// test tests::bench_avg_CascMergeSortTree   ... bench:  18,461,423 ns/iter (+/- 729,043)
// test tests::bench_avg_SimpleMergeSortTree ... bench:  26,054,788 ns/iter (+/- 446,940)

#![feature(test)]
extern crate test;

type Idx = i32;

const NIL : Idx = -1;

#[derive(Clone, Debug)]
struct Entry<T> {
    value : T,
    left_lower  : Idx,
    right_lower : Idx,
}

impl<T> Entry<T> {
    fn new(v : T) -> Self {
        Self { value: v, left_lower: NIL, right_lower: NIL, }
    }
}

trait MergeSortTree<T> {
    fn from_slice(slice : &[T]) -> Self;
    /// Queries how many elements are >= x
    fn query(&self, from : usize, to : usize, x : &T) -> usize;
}

#[derive(Debug)]
struct CascMergeSortTree<T> {
    nodes : Vec<Vec<Entry<T>>>,
}

impl<T : Ord + Clone> CascMergeSortTree<T> {
    fn size(&self) -> usize {
        self.nodes.len() / 2
    }
    fn merge(left : &[Entry<T>], right : &[Entry<T>]) -> Vec<Entry<T>> {
        let mut left_lower  = NIL;
        let mut right_lower = NIL;
        let mut i = 0;
        let mut j = 0;
        let nl = left.len();
        let nr = right.len();
        let mut res = Vec::with_capacity(nl + nr);
        while i < nl && j < nr {
            if left[i].value < right[j].value {
                left_lower = i as Idx;
                res.push(Entry { value: left[i].value.clone(), left_lower, right_lower, });
                i += 1;
            } else {
                right_lower = j as Idx;
                res.push(Entry { value: right[j].value.clone(), left_lower, right_lower, });
                j += 1;
            }
        }
        while i < nl {
            left_lower = i as Idx;
            res.push(Entry { value: left[i].value.clone(), left_lower, right_lower, });
            i += 1;
        }
        while j < nr {
            right_lower = j as Idx;
            res.push(Entry { value: right[j].value.clone(), left_lower, right_lower, });
            j += 1;
        }
        return res;
    }
    fn query_rec(&self, /*incl*/from : usize, /*excl*/to : usize, pp : Idx, /*incl*/node_l : usize, /*excl*/node_r : usize, node_idx : usize, x : &T) -> usize {
        let pp = pp as usize;
        if pp >= self.nodes[node_idx].len() {
            return 0;
        }
        if from <= node_l && node_r <= to {
            return node_r - node_l - pp;
        }
        let node_m = node_l + (node_r - node_l) / 2;
        let left_idx = 2 * node_idx;
        let right_idx = 2 * node_idx + 1;
        let e = &self.nodes[node_idx][pp];
        let mut sum = 0;
        // TODO cleanup
        // if node_l..node_m overlaps with from..=to
        if from <= node_l && node_m <= to || node_l <= from && from < node_m || node_l < to && to <= node_m {
            let lpp = if e.left_lower == NIL {
                0
            } else if self.nodes[left_idx][e.left_lower as usize].value < *x {
                e.left_lower + 1
            } else {
                e.left_lower
            };
            sum += self.query_rec(from, to, lpp, node_l, node_m, left_idx, x);
        }
        // TODO cleanup
        // if node_m..node_r overlaps with from..=to
        if from <= node_m && node_r <= to || node_m <= from && from < node_r || node_m < to && to <= node_r {
            let rpp = if e.right_lower == NIL {
                0
            } else if self.nodes[right_idx][e.right_lower as usize].value < *x {
                e.right_lower + 1
            } else {
                e.right_lower
            };
            sum += self.query_rec(from, to, rpp, node_m, node_r, right_idx, x);
        }
        return sum;
    }
}

impl<T : Ord + Clone> MergeSortTree<T> for CascMergeSortTree<T> {
    fn from_slice(slice : &[T]) -> Self {
        let n = slice.len();
        assert!(n & (n - 1) == 0);
        let mut nodes = vec![Vec::new(); 2 * n];
        for i in 0..n {
            nodes[n + i] = vec![Entry::new(slice[i].clone())];
        }
        for i in (1..n).rev() {
            let left  = 2 * i;
            let right = 2 * i + 1;
            nodes[i] = Self::merge(&nodes[left], &nodes[right]);
        }
        Self { nodes, }
    }
    /// Returns the number of elements greater or equal than x in [from..to]
    fn query(&self, /*incl*/ from : usize, /*excl*/ to : usize, x : &T) -> usize {
        if from == to {
            return 0;
        }
        let pp = self.nodes[1].partition_point(|e| e.value < *x);
        return self.query_rec(from, to, pp as Idx, 0, self.size(), 1, x);
    }
}

#[derive(Debug)]
struct SimpleMergeSortTree<T> {
    nodes : Vec<Vec<T>>,
}

impl<T : Ord + Clone> SimpleMergeSortTree<T> {
    fn merge(left : &[T], right : &[T]) -> Vec<T> {
        let mut i = 0;
        let mut j = 0;
        let nl = left.len();
        let nr = right.len();
        let mut res = Vec::with_capacity(nl + nr);
        while i < nl && j < nr {
            if left[i] < right[j] {
                res.push(left[i].clone());
                i += 1;
            } else {
                res.push(right[j].clone());
                j += 1;
            }
        }
        while i < nl {
            res.push(left[i].clone());
            i += 1;
        }
        while j < nr {
            res.push(right[j].clone());
            j += 1;
        }
        return res;
    }
    fn len(&self) -> usize {
        self.nodes.len() / 2
    }
    fn query_rec(&self, /*incl*/from : usize, /*excl*/to : usize, /*incl*/node_l : usize, /*excl*/node_r : usize, node_idx : usize, x : &T) -> usize {
        if from <= node_l && node_r <= to {
            let pp = self.nodes[node_idx].partition_point(|e| e < x);
            return node_r - node_l - pp;
        }
        let node_overlaps_query_range = node_l <= from && from < node_r || node_l < to && to <= node_r;
        if !node_overlaps_query_range {
            return 0;
        }
        let node_m = node_l + (node_r - node_l) / 2;
        let left_idx  = 2 * node_idx;
        let right_idx = 2 * node_idx + 1;
        return self.query_rec(from, to, node_l, node_m, left_idx, x) + self.query_rec(from, to, node_m, node_r, right_idx, x);
    }
}

use std::fmt::Debug;
impl<T : Debug + Ord + Clone> MergeSortTree<T> for SimpleMergeSortTree<T> {
    fn from_slice(slice : &[T]) -> Self {
        let n = slice.len();
        assert_eq!(n & (n - 1), 0);
        let mut nodes = vec![Vec::new(); 2 * n];
        for i in 0..n {
            nodes[n + i] = vec![slice[i].clone()];
        }
        for i in (1..n).rev() {
            let left = 2 * i;
            let right = 2 * i + 1;
            nodes[i] = Self::merge(&nodes[left], &nodes[right]);
        }
        Self { nodes, }
    }
    /// Returns the number of elements greater or equal than x in [from..to]
    fn query(&self, /*incl*/ from : usize, /*excl*/ to : usize, x : &T) -> usize {
        if from == to {
            return 0;
        }
        return self.query_rec(from, to, 0, self.len(), 1, x);
    }
}

struct Naive<T> {
    data : Vec<T>,
}

impl<T : Ord + Clone> MergeSortTree <T> for Naive<T> {
    fn from_slice(slice : &[T]) -> Self {
        Self { data: Vec::from(slice), }
    }
    fn query(&self, from : usize, to : usize, x : &T) -> usize {
        self.data[from..to].iter().filter(|&d| d >= x).count()
    }
}

fn main() {
    let arr = [1,2,3,4];
    let n = arr.len();
    let cmst = CascMergeSortTree::from_slice(&arr);
    println!("{cmst:?}");
    let nai = Naive::from_slice(&arr);
    println!("{cmst:?}");
    //println!("{}", cmst.query(1, 3, 3));
    let x = 3;
    for i in 0..n {
        for j in i..n {
            dbg!(i, j);
            assert_eq!(nai.query(i, j, &x), cmst.query(i, j, &x));
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    fn my_ilog2(mut n : usize) -> u32 {
        assert_ne!(n, 0);
        let mut ilog = 0;
        loop {
            n >>= 1;
            if n == 0 {
                return ilog;
            }
            ilog += 1;
        }
    }
    fn next_pow_of_2(n : usize) -> usize {
        if n & (n - 1) == 0 {
            return n;
        }
        return usize::pow(2, my_ilog2(n) + 1);
    }
    fn generic_test<Impl : MergeSortTree<usize>>() {
        use rand::*;
        let mut rng = rand::thread_rng();
        let n = 128;
        let mut arr = Vec::new();
        for _ in 0..n {
            arr.push(rng.gen_range(1..=n));
        }
        let cmst = Impl::from_slice(&arr);
        let nai = Naive::from_slice(&arr);
        for i in 0..n {
            for j in i..=n {
                for _ in 0..50 {
                    let x = rng.gen_range(1..=n);
                    assert_eq!(nai.query(i, j, &x), cmst.query(i, j, &x));
                }
            }
        }
    }
    use test::Bencher;
    #[test]
    fn test_SimpleMergeSortTree() {
        generic_test::<SimpleMergeSortTree<usize>>();
    }
    #[test]
    fn test_CascSimpleMergeSortTree() {
        generic_test::<CascMergeSortTree<usize>>();
    }
    fn generic_avg_bench<Impl : MergeSortTree<usize>>(b : &mut Bencher) {
        let n = next_pow_of_2(3 * usize::pow(10, 5));
        let mut arr = Vec::new();
        for i in 0..n {
            arr.push(i);
        }
        let cmst = Impl::from_slice(&arr);
        use std::hint;
        b.iter(||
            for i in (0..=n).step_by(10000) {
                for j in (i..=n).step_by(10001) {
                    for x in (0..=n).step_by(10002) {
                        hint::black_box(cmst.query(i, j, &x));
                    }
                }
            }
        );
    }
    #[bench]
    fn bench_avg_SimpleMergeSortTree(b : &mut Bencher) {
        generic_avg_bench::<SimpleMergeSortTree<usize>>(b);
    }
    #[bench]
    fn bench_avg_CascMergeSortTree(b : &mut Bencher) {
        generic_avg_bench::<CascMergeSortTree<usize>>(b);
    }
}
