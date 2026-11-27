//! A growable bit set of small indices: the many-to-one matcher's sets of
//! viable patterns, which it intersects and restores at every step of the
//! search (a `BTreeSet` there allocates one node per element on every copy).

/// A set of `usize` indices stored as bits.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct BitSet {
    words: Vec<u64>,
}

impl BitSet {
    pub(crate) fn new() -> Self {
        BitSet { words: Vec::new() }
    }

    /// `{0, 1, ..., n - 1}`.
    pub(crate) fn full(n: usize) -> Self {
        let mut words = vec![u64::MAX; n / 64];
        if n % 64 != 0 {
            words.push((1u64 << (n % 64)) - 1);
        }
        BitSet { words }
    }

    pub(crate) fn singleton(i: usize) -> Self {
        let mut s = BitSet::new();
        s.insert(i);
        s
    }

    pub(crate) fn insert(&mut self, i: usize) {
        let w = i / 64;
        if w >= self.words.len() {
            self.words.resize(w + 1, 0);
        }
        self.words[w] |= 1 << (i % 64);
    }

    #[cfg(test)]
    pub(crate) fn contains(&self, i: usize) -> bool {
        self.words.get(i / 64).is_some_and(|w| w & (1 << (i % 64)) != 0)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    pub(crate) fn len(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub(crate) fn is_disjoint(&self, other: &BitSet) -> bool {
        self.words.iter().zip(&other.words).all(|(a, b)| a & b == 0)
    }

    pub(crate) fn intersection(&self, other: &BitSet) -> BitSet {
        let mut words: Vec<u64> = self.words.iter().zip(&other.words).map(|(a, b)| a & b).collect();
        while words.last() == Some(&0) {
            words.pop();
        }
        BitSet { words }
    }

    pub(crate) fn union_with(&mut self, other: &BitSet) {
        if other.words.len() > self.words.len() {
            self.words.resize(other.words.len(), 0);
        }
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a |= b;
        }
    }

    /// Remove the elements of `other`.
    pub(crate) fn difference_with(&mut self, other: &BitSet) {
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a &= !b;
        }
    }

    /// The elements in increasing order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(i, &w)| {
            let mut w = w;
            std::iter::from_fn(move || {
                if w == 0 {
                    return None;
                }
                let bit = w.trailing_zeros() as usize;
                w &= w - 1;
                Some(i * 64 + bit)
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::BitSet;

    #[test]
    fn operations() {
        let mut a = BitSet::full(130);
        assert_eq!(a.len(), 130);
        assert!(a.contains(129) && !a.contains(130));
        let mut b = BitSet::singleton(3);
        b.insert(70);
        b.insert(200);
        assert_eq!(a.intersection(&b).iter().collect::<Vec<_>>(), vec![3, 70]);
        assert!(!a.is_disjoint(&b));
        a.difference_with(&b);
        assert!(!a.contains(3) && !a.contains(70) && a.contains(4));
        assert_eq!(a.len(), 128);
        let c = BitSet::singleton(5).intersection(&BitSet::singleton(64));
        assert!(c.is_empty() && c.iter().next().is_none());
        let mut d = BitSet::new();
        d.union_with(&b);
        assert_eq!(d.iter().collect::<Vec<_>>(), vec![3, 70, 200]);
        assert!(BitSet::singleton(2).is_disjoint(&BitSet::singleton(200)));
    }
}
