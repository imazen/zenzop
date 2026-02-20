use alloc::vec::Vec;
use core::cmp::{self, Ordering};

// Bounded package merge algorithm, based on the paper
// "A Fast and Space-Economical Algorithm for Length-Limited Coding
// Jyrki Katajainen, Alistair Moffat, Andrew Turpin".

const NONE: u32 = u32::MAX;

struct Thing {
    nodes: Vec<Node>,
    leaves: Vec<Leaf>,
    lists: [List; 15],
}

struct Node {
    weight: usize,
    count: usize,
    tail: u32, // Index into nodes Vec. NONE = no tail.
}

struct Leaf {
    weight: usize,
    count: usize,
}
impl PartialEq for Leaf {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight
    }
}
impl Eq for Leaf {}
impl Ord for Leaf {
    fn cmp(&self, other: &Self) -> Ordering {
        self.weight.cmp(&other.weight)
    }
}
impl PartialOrd for Leaf {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy)]
struct List {
    lookahead0: u32, // Index into nodes Vec.
    lookahead1: u32, // Index into nodes Vec.
}

/// Calculates the bitlengths for the Huffman tree, based on the counts of each
/// symbol.
pub fn length_limited_code_lengths(frequencies: &[usize], max_bits: usize) -> Vec<u32> {
    let num_freqs = frequencies.len();
    assert!(num_freqs <= 288);

    // Count used symbols and place them in the leaves.
    let mut leaves = frequencies
        .iter()
        .enumerate()
        .filter_map(|(i, &freq)| {
            (freq != 0).then_some(Leaf {
                weight: freq,
                count: i,
            })
        })
        .collect::<Vec<_>>();

    let num_symbols = leaves.len();

    // Short circuit some special cases

    // TODO:
    // if ((1 << maxbits) < numsymbols) {
    //   free(leaves);
    //   return 1;  /* Error, too few maxbits to represent symbols. */
    // }

    if num_symbols <= 2 {
        // The symbols for the non-zero frequencies can be represented
        // with zero or one bits.
        let mut bit_lengths = vec![0; num_freqs];
        for i in 0..num_symbols {
            bit_lengths[leaves[i].count] = 1;
        }
        return bit_lengths;
    }

    // Sort the leaves from least frequent to most frequent.
    leaves.sort();

    let max_bits = cmp::min(num_symbols - 1, max_bits);
    assert!(max_bits <= 15);

    let capacity = max_bits * 2 * num_symbols;
    let mut nodes = Vec::with_capacity(capacity);

    nodes.push(Node {
        weight: leaves[0].weight,
        count: 1,
        tail: NONE,
    });
    nodes.push(Node {
        weight: leaves[1].weight,
        count: 2,
        tail: NONE,
    });

    let lists = [List {
        lookahead0: 0,
        lookahead1: 1,
    }; 15];

    let mut thing = Thing {
        nodes,
        leaves,
        lists,
    };

    // In the last list, 2 * num_symbols - 2 active chains need to be created. Two
    // are already created in the initialization. Each boundary_pm run creates one.
    let num_boundary_pm_runs = 2 * num_symbols - 4;
    for _ in 0..num_boundary_pm_runs - 1 {
        thing.boundary_pm(max_bits - 1);
    }

    thing.boundary_pm_final(max_bits - 1);

    thing.extract_bit_lengths(max_bits, num_freqs)
}

impl Thing {
    fn boundary_pm(&mut self, index: usize) {
        let num_symbols = self.leaves.len();

        let last_count = self.nodes[self.lists[index].lookahead1 as usize].count;

        if index == 0 && last_count >= num_symbols {
            return;
        }

        self.lists[index].lookahead0 = self.lists[index].lookahead1;

        if index == 0 {
            // New leaf node in list 0.
            let tail = self.nodes[self.lists[index].lookahead0 as usize].tail;
            let idx = self.nodes.len() as u32;
            self.nodes.push(Node {
                weight: self.leaves[last_count].weight,
                count: last_count + 1,
                tail,
            });
            self.lists[index].lookahead1 = idx;
        } else {
            let weight_sum = {
                let previous_list = &self.lists[index - 1];
                self.nodes[previous_list.lookahead0 as usize].weight
                    + self.nodes[previous_list.lookahead1 as usize].weight
            };

            if last_count < num_symbols && weight_sum > self.leaves[last_count].weight {
                // New leaf inserted in list, so count is incremented.
                let tail = self.nodes[self.lists[index].lookahead0 as usize].tail;
                let idx = self.nodes.len() as u32;
                self.nodes.push(Node {
                    weight: self.leaves[last_count].weight,
                    count: last_count + 1,
                    tail,
                });
                self.lists[index].lookahead1 = idx;
            } else {
                let tail = self.lists[index - 1].lookahead1;
                let idx = self.nodes.len() as u32;
                self.nodes.push(Node {
                    weight: weight_sum,
                    count: last_count,
                    tail,
                });
                self.lists[index].lookahead1 = idx;

                // Two lookahead chains of previous list used up, create new ones.
                self.boundary_pm(index - 1);
                self.boundary_pm(index - 1);
            }
        }
    }

    fn boundary_pm_final(&mut self, index: usize) {
        let num_symbols = self.leaves.len();

        // Count of last chain of list.
        let last_count = self.nodes[self.lists[index].lookahead1 as usize].count;

        let weight_sum = {
            let previous_list = &self.lists[index - 1];
            self.nodes[previous_list.lookahead0 as usize].weight
                + self.nodes[previous_list.lookahead1 as usize].weight
        };

        if last_count < num_symbols && weight_sum > self.leaves[last_count].weight {
            let tail = self.nodes[self.lists[index].lookahead1 as usize].tail;
            let idx = self.nodes.len() as u32;
            self.nodes.push(Node {
                weight: 0,
                count: last_count + 1,
                tail,
            });
            self.lists[index].lookahead1 = idx;
        } else {
            let node_idx = self.lists[index].lookahead1 as usize;
            self.nodes[node_idx].tail = self.lists[index - 1].lookahead1;
        }
    }

    fn extract_bit_lengths(&self, max_bits: usize, num_freqs: usize) -> Vec<u32> {
        let mut counts = [0; 16];
        let mut end = 16;
        let mut ptr = 15;
        let mut value = 1;

        let mut node_idx = self.lists[max_bits - 1].lookahead1 as usize;

        end -= 1;
        counts[end] = self.nodes[node_idx].count;

        let mut tail = self.nodes[node_idx].tail;
        while tail != NONE {
            end -= 1;
            node_idx = tail as usize;
            counts[end] = self.nodes[node_idx].count;
            tail = self.nodes[node_idx].tail;
        }

        let mut val = counts[15];

        let mut bit_lengths = vec![0; num_freqs];

        while ptr >= end {
            while val > counts[ptr - 1] {
                bit_lengths[self.leaves[val - 1].count] = value;
                val -= 1;
            }
            ptr -= 1;
            value += 1;
        }

        bit_lengths
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_from_paper_3() {
        let input = [1, 1, 5, 7, 10, 14];
        let output = length_limited_code_lengths(&input, 3);
        let answer = vec![3, 3, 3, 3, 2, 2];
        assert_eq!(output, answer);
    }

    #[test]
    fn test_from_paper_4() {
        let input = [1, 1, 5, 7, 10, 14];
        let output = length_limited_code_lengths(&input, 4);
        let answer = vec![4, 4, 3, 2, 2, 2];
        assert_eq!(output, answer);
    }

    #[test]
    fn max_bits_7() {
        let input = [252, 0, 1, 6, 9, 10, 6, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let output = length_limited_code_lengths(&input, 7);
        let answer = vec![1, 0, 6, 4, 3, 3, 3, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(output, answer);
    }

    #[test]
    fn max_bits_15() {
        let input = [
            0, 0, 0, 0, 0, 0, 18, 0, 6, 0, 12, 2, 14, 9, 27, 15, 23, 15, 17, 8, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ];
        let output = length_limited_code_lengths(&input, 15);
        let answer = vec![
            0, 0, 0, 0, 0, 0, 3, 0, 5, 0, 4, 6, 4, 4, 3, 4, 3, 3, 3, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        ];
        assert_eq!(output, answer);
    }

    #[test]
    fn no_frequencies() {
        let input = [0, 0, 0, 0, 0];
        let output = length_limited_code_lengths(&input, 7);
        let answer = vec![0, 0, 0, 0, 0];
        assert_eq!(output, answer);
    }

    #[test]
    fn only_one_frequency() {
        let input = [0, 10, 0];
        let output = length_limited_code_lengths(&input, 7);
        let answer = vec![0, 1, 0];
        assert_eq!(output, answer);
    }

    #[test]
    fn only_two_frequencies() {
        let input = [0, 0, 0, 0, 252, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let output = length_limited_code_lengths(&input, 7);
        let answer = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(output, answer);
    }
}
