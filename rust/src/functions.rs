//! Substitution and rewriting.
//!
//! Port of `omnimatch/functions.py`.

use std::fmt;
use std::sync::Arc;

use crate::expressions::{Expr, ExprError, Pattern};
use crate::matching::one_to_one::{first_match, MatchError};
use crate::substitution::{Replacement, Substitution};

/// Errors from the rewriting functions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RewriteError {
    Expr(ExprError),
    Match(MatchError),
    /// A rule replaced the root expression with a sequence or multiset, which
    /// cannot be matched further.
    NonSingleRoot,
}

impl fmt::Display for RewriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RewriteError::Expr(e) => write!(f, "{e}"),
            RewriteError::Match(e) => write!(f, "{e}"),
            RewriteError::NonSingleRoot => write!(f, "the root expression was replaced by a sequence"),
        }
    }
}

impl std::error::Error for RewriteError {}

impl From<ExprError> for RewriteError {
    fn from(e: ExprError) -> Self {
        RewriteError::Expr(e)
    }
}

impl From<MatchError> for RewriteError {
    fn from(e: MatchError) -> Self {
        RewriteError::Match(e)
    }
}

/// Replace the variables in `expression` using `substitution`.
///
/// A variable bound to a sequence is spliced into the surrounding operation; a
/// multiset is spliced in sorted order. The result is a sequence only if
/// `expression` itself is a variable bound to one.
pub fn substitute(expression: &Expr, substitution: &Substitution) -> Result<Replacement, ExprError> {
    Ok(substitute_rec(expression, substitution, None)?.0)
}

/// Like [`substitute`], but multisets are spliced in the order given by
/// `compare` instead of the canonical one (Python: the `sort_key` argument).
pub fn substitute_sorted_by(
    expression: &Expr,
    substitution: &Substitution,
    compare: &dyn Fn(&Expr, &Expr) -> std::cmp::Ordering,
) -> Result<Replacement, ExprError> {
    Ok(substitute_rec(expression, substitution, Some(compare))?.0)
}

type Compare<'c> = Option<&'c dyn Fn(&Expr, &Expr) -> std::cmp::Ordering>;

/// Append `replacement` to `operands`, splicing sequences and multisets.
fn splice(operands: &mut Vec<Expr>, replacement: Replacement, compare: Compare<'_>) -> Result<(), ExprError> {
    match replacement {
        Replacement::Single(e) => operands.push(e),
        Replacement::Sequence(v) => operands.extend(v),
        Replacement::Multiset(m) => {
            // Multiset iteration is already in canonical order.
            let mut items: Vec<Expr> = m.iter().cloned().collect();
            if let Some(compare) = compare {
                items.sort_by(|a, b| compare(a, b));
            }
            operands.extend(items);
        }
        Replacement::Head(h) => {
            return Err(ExprError::InvalidPosition(format!("cannot use the head {} as an operand", h.name)))
        }
    }
    Ok(())
}

fn substitute_rec(
    expression: &Expr,
    substitution: &Substitution,
    compare: Compare<'_>,
) -> Result<(Replacement, bool), ExprError> {
    if let Some(replacement) = expression.variable_name().and_then(|v| substitution.get(v)) {
        return Ok((replacement.clone(), true));
    }
    if let Some((head, operands)) = expression.as_operation() {
        let mut any_replaced = false;
        let mut new_operands = Vec::with_capacity(operands.len());
        for operand in operands {
            let (result, replaced) = substitute_rec(operand, substitution, compare)?;
            any_replaced |= replaced;
            splice(&mut new_operands, result, compare)?;
        }
        // A wildcard head whose variable is bound to a head takes that head.
        let bound_head = match head.head_variable.as_deref().and_then(|v| substitution.get(v)) {
            Some(Replacement::Head(h)) if head.wildcard => Some(h.clone()),
            _ => None,
        };
        any_replaced |= bound_head.is_some();
        if any_replaced {
            let head = bound_head.unwrap_or_else(|| head.clone());
            let rebuilt = Expr::try_operation(head, new_operands, expression.variable_name())?;
            return Ok((Replacement::Single(rebuilt), true));
        }
    }
    Ok((Replacement::Single(expression.clone()), false))
}

/// Replace the subexpression at `position` with `replacement`. A sequence
/// replacement is spliced into the parent operation.
pub fn replace(expression: &Expr, position: &[usize], replacement: Replacement) -> Result<Replacement, ExprError> {
    let Some((&pos, rest)) = position.split_first() else { return Ok(replacement) };
    let Some((head, operands)) = expression.as_operation() else {
        return Err(ExprError::InvalidPosition(format!("Invalid position {position:?} for expression {expression}")));
    };
    if pos >= operands.len() {
        return Err(ExprError::InvalidPosition(format!(
            "Position {position:?} out of range for expression {expression}"
        )));
    }
    let sub = replace(&operands[pos], rest, replacement)?;
    let mut new_operands: Vec<Expr> = operands[..pos].to_vec();
    splice(&mut new_operands, sub, None)?;
    new_operands.extend_from_slice(&operands[pos + 1..]);
    Ok(Replacement::Single(Expr::try_operation(head.clone(), new_operands, expression.variable_name())?))
}

/// Replace several positions at once. Unlike calling [`replace`] repeatedly,
/// positions refer to the original expression even when earlier replacements
/// are sequences that shift later operands.
///
/// ```
/// use omnimatch::*;
/// let f = Head::new("f", Arity::Variadic).unwrap();
/// let (a, b, c) = (Expr::atom("a"), Expr::atom("b"), Expr::atom("c"));
/// let expr = f.call(vec![a.clone(), b]);
/// let r = replace_many(&expr, vec![(vec![0], Replacement::Sequence(vec![c.clone(), c])), (vec![1], a.into())]);
/// assert_eq!(r.unwrap().to_string(), "f(c, c, a)");
/// ```
pub fn replace_many(
    expression: &Expr,
    mut replacements: Vec<(Vec<usize>, Replacement)>,
) -> Result<Replacement, ExprError> {
    if replacements.is_empty() {
        return Ok(Replacement::Single(expression.clone()));
    }
    replacements.sort_by(|x, y| x.0.cmp(&y.0));
    if replacements[0].0.is_empty() {
        if replacements.len() > 1 {
            return Err(ExprError::InvalidPosition(format!(
                "Cannot replace child positions for expression {expression}, got {:?}",
                replacements[1..].iter().map(|r| &r.0).collect::<Vec<_>>()
            )));
        }
        return Ok(replacements.pop().expect("one replacement").1);
    }
    if replacements.len() == 1 {
        let (position, replacement) = replacements.pop().expect("one replacement");
        return replace(expression, &position, replacement);
    }
    let Some((head, operands)) = expression.as_operation() else {
        return Err(ExprError::InvalidPosition(format!("Invalid replacements for expression {expression}")));
    };
    let mut new_operands = Vec::new();
    let mut last_index = 0;
    let mut iter = replacements.into_iter().peekable();
    while let Some((position, replacement)) = iter.next() {
        let index = position[0];
        if index >= operands.len() {
            return Err(ExprError::InvalidPosition(format!(
                "Position {position:?} out of range for expression {expression}"
            )));
        }
        let mut group = vec![(position[1..].to_vec(), replacement)];
        while let Some((p, _)) = iter.peek() {
            if p[0] != index {
                break;
            }
            let (p, r) = iter.next().expect("peeked");
            group.push((p[1..].to_vec(), r));
        }
        new_operands.extend_from_slice(&operands[last_index..index]);
        let replaced = if group.len() == 1 {
            let (p, r) = group.pop().expect("one replacement");
            replace(&operands[index], &p, r)?
        } else {
            replace_many(&operands[index], group)?
        };
        splice(&mut new_operands, replaced, None)?;
        last_index = index + 1;
    }
    new_operands.extend_from_slice(&operands[last_index..]);
    Ok(Replacement::Single(Expr::try_operation(head.clone(), new_operands, expression.variable_name())?))
}

/// The replacement callback of a [`ReplacementRule`].
pub type ReplacementFn = Arc<dyn Fn(&Substitution) -> Replacement + Send + Sync>;

/// A pattern and the callback computing its replacement from the match.
#[derive(Clone)]
pub struct ReplacementRule {
    pub pattern: Pattern,
    pub replacement: ReplacementFn,
}

impl ReplacementRule {
    pub fn new(pattern: Pattern, replacement: impl Fn(&Substitution) -> Replacement + Send + Sync + 'static) -> Self {
        ReplacementRule { pattern, replacement: Arc::new(replacement) }
    }
}

fn single_root(r: Replacement) -> Result<Expr, RewriteError> {
    match r {
        Replacement::Single(e) => Ok(e),
        _ => Err(RewriteError::NonSingleRoot),
    }
}

/// Repeatedly apply the first matching rule at the first (pre-order) matching
/// position until nothing matches or `max_count` rewrites were done.
pub fn replace_all(
    expression: &Expr,
    rules: &[ReplacementRule],
    max_count: Option<usize>,
) -> Result<Replacement, RewriteError> {
    let mut expression = expression.clone();
    let mut count = 0;
    'outer: while max_count.map_or(true, |m| count < m) {
        for (subexpr, pos) in expression.preorder_iter() {
            for rule in rules {
                if let Some(subst) = first_match(&subexpr, &rule.pattern)? {
                    let result = replace(&expression, &pos, (rule.replacement)(&subst))?;
                    count += 1;
                    if pos.is_empty() && !matches!(result, Replacement::Single(_)) {
                        return Ok(result);
                    }
                    expression = single_root(result)?;
                    continue 'outer;
                }
            }
        }
        break;
    }
    Ok(Replacement::Single(expression))
}

/// Apply the rules bottom-up (operands first) until no rule matches anywhere.
pub fn replace_all_post_order(expression: &Expr, rules: &[ReplacementRule]) -> Result<Expr, RewriteError> {
    Ok(post_order_rec(expression.clone(), rules)?.0)
}

fn post_order_rec(mut expression: Expr, rules: &[ReplacementRule]) -> Result<(Expr, bool), RewriteError> {
    let mut any_replaced = false;
    loop {
        if let Some((head, operands)) = expression.as_operation() {
            let mut changed = false;
            let mut new_operands = Vec::with_capacity(operands.len());
            for operand in operands {
                let (e, r) = post_order_rec(operand.clone(), rules)?;
                changed |= r;
                new_operands.push(e);
            }
            if changed {
                expression = Expr::try_operation(head.clone(), new_operands, expression.variable_name())?;
                any_replaced = true;
            }
        }
        let mut replaced = false;
        for rule in rules {
            if let Some(subst) = first_match(&expression, &rule.pattern)? {
                expression = single_root((rule.replacement)(&subst))?;
                replaced = true;
                any_replaced = true;
                break;
            }
        }
        if !replaced {
            return Ok((expression, any_replaced));
        }
    }
}
