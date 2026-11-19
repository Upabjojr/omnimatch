//! A small ordered multiset, replacing the Python `multiset` dependency.
//!
//! Elements are kept in a `BTreeMap` from element to multiplicity, so iteration
//! (and therefore `Display`) is always in sorted order.

use std::collections::btree_map;
use std::collections::BTreeMap;
use std::fmt;

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Multiset<T: Ord> {
    counts: BTreeMap<T, usize>,
    len: usize,
}

impl<T: Ord> Default for Multiset<T> {
    fn default() -> Self {
        Multiset { counts: BTreeMap::new(), len: 0 }
    }
}

impl<T: Ord + Clone> Multiset<T> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Total number of elements, counting multiplicity.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of distinct elements.
    pub fn distinct_len(&self) -> usize {
        self.counts.len()
    }

    pub fn count(&self, item: &T) -> usize {
        self.counts.get(item).copied().unwrap_or(0)
    }

    pub fn contains(&self, item: &T) -> bool {
        self.counts.contains_key(item)
    }

    pub fn add(&mut self, item: T) {
        self.add_n(item, 1);
    }

    pub fn add_n(&mut self, item: T, n: usize) {
        if n == 0 {
            return;
        }
        *self.counts.entry(item).or_insert(0) += n;
        self.len += n;
    }

    /// Set the multiplicity of `item` (0 removes it).
    pub fn set_count(&mut self, item: T, n: usize) {
        let old = if n == 0 { self.counts.remove(&item) } else { self.counts.insert(item, n) };
        self.len = self.len + n - old.unwrap_or(0);
    }

    /// Remove every occurrence of `item`.
    pub fn remove_all(&mut self, item: &T) {
        if let Some(n) = self.counts.remove(item) {
            self.len -= n;
        }
    }

    /// Iterate over `(element, multiplicity)` pairs in sorted order.
    pub fn items(&self) -> btree_map::Iter<'_, T, usize> {
        self.counts.iter()
    }

    pub fn distinct_elements(&self) -> btree_map::Keys<'_, T, usize> {
        self.counts.keys()
    }

    /// Iterate over all elements, repeating each according to its multiplicity.
    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        self.counts.iter().flat_map(|(k, &n)| std::iter::repeat(k).take(n))
    }

    /// `self <= other` in the multiset sense.
    pub fn is_subset(&self, other: &Self) -> bool {
        self.len <= other.len && self.counts.iter().all(|(k, &n)| other.count(k) >= n)
    }

    /// Multiset difference (saturating at zero).
    pub fn difference(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (k, &n) in other.items() {
            let have = result.count(k);
            result.set_count(k.clone(), have.saturating_sub(n));
        }
        result
    }

    /// Multiset sum.
    pub fn combine(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (k, &n) in other.items() {
            result.add_n(k.clone(), n);
        }
        result
    }

    /// Every multiplicity multiplied by `factor`.
    pub fn times(&self, factor: usize) -> Self {
        let mut result = Self::new();
        for (k, &n) in self.items() {
            result.add_n(k.clone(), n * factor);
        }
        result
    }

    /// Copy without one occurrence of `item`.
    pub fn without_one(&self, item: &T) -> Self {
        let mut result = self.clone();
        let have = result.count(item);
        if have > 0 {
            result.set_count(item.clone(), have - 1);
        }
        result
    }
}

impl<T: Ord + Clone> FromIterator<T> for Multiset<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut m = Multiset::new();
        for item in iter {
            m.add(item);
        }
        m
    }
}

impl<T: Ord + Clone + fmt::Display> fmt::Display for Multiset<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        for (i, item) in self.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{item}")?;
        }
        write!(f, "}}")
    }
}

impl<T: Ord + Clone + fmt::Debug> fmt::Debug for Multiset<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_ops() {
        let a: Multiset<char> = "aaabbc".chars().collect();
        assert_eq!(a.len(), 6);
        assert_eq!(a.count(&'a'), 3);
        let b: Multiset<char> = "ab".chars().collect();
        assert!(b.is_subset(&a));
        assert!(!a.is_subset(&b));
        let d = a.difference(&b);
        assert_eq!(d, "aabc".chars().collect());
        assert_eq!(b.times(2), "aabb".chars().collect());
        assert_eq!(b.combine(&b), "aabb".chars().collect());
        assert_eq!(a.without_one(&'c'), "aaabb".chars().collect());
        let mut e = a.clone();
        e.set_count('a', 0);
        assert_eq!(e.len(), 3);
        assert_eq!(format!("{}", b), "{a, b}");
    }
}
