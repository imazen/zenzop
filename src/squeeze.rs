//! The squeeze functions do enhanced LZ77 compression by optimal parsing with a
//! cost model, rather than greedily choosing the longest length or using a single
//! step of lazy matching like regular implementations.
//!
//! Since the cost model is based on the Huffman tree that can only be calculated
//! after the LZ77 data is generated, there is a chicken and egg problem, and
//! multiple runs are done with updated cost models to converge to a better
//! solution.

use alloc::vec::Vec;
use core::cmp;

#[cfg(feature = "std")]
use log::{debug, trace};

use crate::{
    cache::Cache,
    deflate::{calculate_block_size, BlockType},
    hash::ZopfliHash,
    lz77::{find_longest_match, LitLen, Lz77Store},
    symbols::{
        get_dist_extra_bits, get_dist_symbol, get_dist_symbol_extra_bits, get_length_extra_bits,
        get_length_symbol, get_length_symbol_extra_bits,
    },
    util::{ZOPFLI_MAX_MATCH, ZOPFLI_NUM_D, ZOPFLI_NUM_LL, ZOPFLI_WINDOW_MASK, ZOPFLI_WINDOW_SIZE},
};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)] // False-positive
use crate::math::F64MathExt;

/// Cost model which should exactly match fixed tree.
fn get_cost_fixed(litlen: usize, dist: u16) -> f64 {
    let result = if dist == 0 {
        if litlen <= 143 {
            8
        } else {
            9
        }
    } else {
        let dbits = get_dist_extra_bits(dist);
        let lbits = get_length_extra_bits(litlen);
        let lsym = get_length_symbol(litlen);
        // Every dist symbol has length 5.
        7 + u32::from(lsym > 279) + 5 + dbits + lbits
    };
    f64::from(result)
}

/// Precomputed cost lookup tables for the stat-based cost model.
/// Eliminates per-symbol f64 math in the inner DP loop.
struct CostModel {
    /// Cost of literal byte i (entropy only).
    ll_literal: [f32; 256],
    /// Cost of match length i (entropy + extra bits). Only indices 3..=258 are valid.
    ll_length: [f32; ZOPFLI_MAX_MATCH + 1],
    /// Cost of distance symbol d (entropy + extra bits).
    d_cost: [f32; ZOPFLI_NUM_D],
}

impl CostModel {
    fn from_stats(stats: &SymbolStats) -> Self {
        let mut ll_literal = [0.0f32; 256];
        for (i, cost) in ll_literal.iter_mut().enumerate() {
            *cost = stats.ll_symbols[i] as f32;
        }

        let mut ll_length = [0.0f32; ZOPFLI_MAX_MATCH + 1];
        for (i, cost) in ll_length.iter_mut().enumerate().skip(3) {
            let lsym = get_length_symbol(i);
            *cost = (stats.ll_symbols[lsym] + f64::from(get_length_symbol_extra_bits(lsym))) as f32;
        }

        let mut d_cost = [0.0f32; ZOPFLI_NUM_D];
        // Only 30 of 32 dist symbols are used in DEFLATE
        for (dsym, cost) in d_cost.iter_mut().enumerate().take(30) {
            *cost = (stats.d_symbols[dsym] + f64::from(get_dist_symbol_extra_bits(dsym))) as f32;
        }

        Self {
            ll_literal,
            ll_length,
            d_cost,
        }
    }

    #[inline(always)]
    fn cost(&self, litlen: usize, dist: u16) -> f64 {
        if dist == 0 {
            f64::from(self.ll_literal[litlen])
        } else {
            f64::from(self.ll_length[litlen])
                + f64::from(self.d_cost[get_dist_symbol(dist) as usize])
        }
    }
}

#[derive(Default)]
struct RanState {
    m_w: u32,
    m_z: u32,
}

impl RanState {
    const fn new() -> Self {
        Self { m_w: 1, m_z: 2 }
    }

    /// Get random number: "Multiply-With-Carry" generator of G. Marsaglia
    fn random_marsaglia(&mut self) -> u32 {
        self.m_z = 36969 * (self.m_z & 65535) + (self.m_z >> 16);
        self.m_w = 18000 * (self.m_w & 65535) + (self.m_w >> 16);
        (self.m_z << 16).wrapping_add(self.m_w) // 32-bit result.
    }
}

#[derive(Copy, Clone)]
struct SymbolStats {
    /* The literal and length symbols. */
    litlens: [usize; ZOPFLI_NUM_LL],
    /* The 32 unique dist symbols, not the 32768 possible dists. */
    dists: [usize; ZOPFLI_NUM_D],

    /* Length of each lit/len symbol in bits. */
    ll_symbols: [f64; ZOPFLI_NUM_LL],
    /* Length of each dist symbol in bits. */
    d_symbols: [f64; ZOPFLI_NUM_D],
}

impl Default for SymbolStats {
    fn default() -> Self {
        Self {
            litlens: [0; ZOPFLI_NUM_LL],
            dists: [0; ZOPFLI_NUM_D],
            ll_symbols: [0.0; ZOPFLI_NUM_LL],
            d_symbols: [0.0; ZOPFLI_NUM_D],
        }
    }
}

impl SymbolStats {
    fn randomize_stat_freqs(&mut self, state: &mut RanState) {
        fn randomize_freqs(freqs: &mut [usize], state: &mut RanState) {
            let n = freqs.len();
            let mut i = 0;
            let end = n;

            while i < end {
                if (state.random_marsaglia() >> 4) % 3 == 0 {
                    let index = state.random_marsaglia() as usize % n;
                    freqs[i] = freqs[index];
                }
                i += 1;
            }
        }
        randomize_freqs(&mut self.litlens, state);
        randomize_freqs(&mut self.dists, state);
        self.litlens[256] = 1; // End symbol.
    }

    /// Calculates the entropy of each symbol, based on the counts of each symbol. The
    /// result is similar to the result of `length_limited_code_lengths`, but with the
    /// actual theoretical bit lengths according to the entropy. Since the resulting
    /// values are fractional, they cannot be used to encode the tree specified by
    /// DEFLATE.
    fn calculate_entropy(&mut self) {
        fn calculate_and_store_entropy(count: &[usize], bitlengths: &mut [f64]) {
            let n = count.len();

            let sum = count.iter().sum();

            let log2sum = (if sum == 0 { n } else { sum } as f64).log2();

            for i in 0..n {
                // When the count of the symbol is 0, but its cost is requested anyway, it
                // means the symbol will appear at least once anyway, so give it the cost as if
                // its count is 1.
                if count[i] == 0 {
                    bitlengths[i] = log2sum;
                } else {
                    bitlengths[i] = log2sum - (count[i] as f64).log2();
                }
            }
        }

        calculate_and_store_entropy(&self.litlens, &mut self.ll_symbols);
        calculate_and_store_entropy(&self.dists, &mut self.d_symbols);
    }

    /// Appends the symbol statistics from the store.
    fn get_statistics(&mut self, store: &Lz77Store) {
        for &litlen in &store.litlens {
            match litlen {
                LitLen::Literal(lit) => self.litlens[lit as usize] += 1,
                LitLen::LengthDist(len, dist) => {
                    self.litlens[get_length_symbol(len as usize)] += 1;
                    self.dists[get_dist_symbol(dist) as usize] += 1;
                }
            }
        }
        self.litlens[256] = 1; /* End symbol. */

        self.calculate_entropy();
    }

    fn clear_freqs(&mut self) {
        self.litlens = [0; ZOPFLI_NUM_LL];
        self.dists = [0; ZOPFLI_NUM_D];
    }
}

fn add_weighed_stat_freqs(
    stats1: &SymbolStats,
    w1: f64,
    stats2: &SymbolStats,
    w2: f64,
) -> SymbolStats {
    let mut result = SymbolStats::default();

    for i in 0..ZOPFLI_NUM_LL {
        result.litlens[i] =
            (stats1.litlens[i] as f64 * w1 + stats2.litlens[i] as f64 * w2) as usize;
    }
    for i in 0..ZOPFLI_NUM_D {
        result.dists[i] = (stats1.dists[i] as f64 * w1 + stats2.dists[i] as f64 * w2) as usize;
    }
    result.litlens[256] = 1; // End symbol.
    result
}

/// Finds the minimum possible cost this cost model can return for valid length and
/// distance symbols.
fn get_cost_model_min_cost<F: Fn(usize, u16) -> f64>(costmodel: F) -> f64 {
    let mut bestlength = 0; // length that has lowest cost in the cost model
    let mut bestdist = 0; // distance that has lowest cost in the cost model

    // Table of distances that have a different distance symbol in the deflate
    // specification. Each value is the first distance that has a new symbol. Only
    // different symbols affect the cost model so only these need to be checked.
    // See RFC 1951 section 3.2.5. Compressed blocks (length and distance codes).

    const DSYMBOLS: [u16; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
        2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
    ];

    let mut mincost = f64::INFINITY;
    for i in 3..259 {
        let c = costmodel(i, 1);
        if c < mincost {
            bestlength = i;
            mincost = c;
        }
    }

    mincost = f64::INFINITY;
    for dsym in DSYMBOLS {
        let c = costmodel(3, dsym);
        if c < mincost {
            bestdist = dsym;
            mincost = c;
        }
    }
    costmodel(bestlength, bestdist)
}

/// Performs the forward pass for "squeeze". Gets the most optimal length to reach
/// every byte from a previous byte, using cost calculations.
/// `s`: the `ZopfliBlockState`
/// `in_data`: the input data array
/// `instart`: where to start
/// `inend`: where to stop (not inclusive)
/// `costmodel`: function to calculate the cost of some lit/len/dist pair.
/// `length_array`: output array of size `(inend - instart)` which will receive the best
///     length to reach this byte from a previous byte.
/// returns the cost that was, according to the `costmodel`, needed to get to the end.
#[allow(clippy::too_many_arguments)]
fn get_best_lengths<F: Fn(usize, u16) -> f64, C: Cache>(
    lmc: &mut C,
    in_data: &[u8],
    instart: usize,
    inend: usize,
    costmodel: F,
    h: &mut ZopfliHash,
    costs: &mut Vec<f32>,
    length_array: &mut Vec<u16>,
    dist_array: &mut Vec<u16>,
    sublen: &mut Vec<u16>,
) -> f64 {
    // Best cost to get here so far.
    let blocksize = inend - instart;
    length_array.clear();
    length_array.resize(blocksize + 1, 0);
    dist_array.clear();
    dist_array.resize(blocksize + 1, 0);
    if instart == inend {
        return 0.0;
    }
    let windowstart = instart.saturating_sub(ZOPFLI_WINDOW_SIZE);

    h.reset();
    let arr = &in_data[..inend];
    h.warmup(arr, windowstart, inend);
    for i in windowstart..instart {
        h.update(arr, i);
    }

    costs.resize(blocksize + 1, 0.0);
    for cost in costs.iter_mut().take(blocksize + 1).skip(1) {
        *cost = f32::INFINITY;
    }
    costs[0] = 0.0; /* Because it's the start. */

    let mut i = instart;
    let mut leng;
    let mut longest_match;
    sublen.resize(ZOPFLI_MAX_MATCH + 1, 0);
    let mincost = get_cost_model_min_cost(&costmodel);
    while i < inend {
        let mut j = i - instart; // Index in the costs array and length_array.
        h.update(arr, i);

        // If we're in a long repetition of the same character and have more than
        // ZOPFLI_MAX_MATCH characters before and after our position.
        if h.same[i & ZOPFLI_WINDOW_MASK] > ZOPFLI_MAX_MATCH as u16 * 2
            && i > instart + ZOPFLI_MAX_MATCH + 1
            && i + ZOPFLI_MAX_MATCH * 2 + 1 < inend
            && h.same[(i - ZOPFLI_MAX_MATCH) & ZOPFLI_WINDOW_MASK] > ZOPFLI_MAX_MATCH as u16
        {
            let symbolcost = costmodel(ZOPFLI_MAX_MATCH, 1);
            // Set the length to reach each one to ZOPFLI_MAX_MATCH, and the cost to
            // the cost corresponding to that length. Doing this, we skip
            // ZOPFLI_MAX_MATCH values to avoid calling ZopfliFindLongestMatch.

            for _ in 0..ZOPFLI_MAX_MATCH {
                costs[j + ZOPFLI_MAX_MATCH] = costs[j] + symbolcost as f32;
                length_array[j + ZOPFLI_MAX_MATCH] = ZOPFLI_MAX_MATCH as u16;
                dist_array[j + ZOPFLI_MAX_MATCH] = 1;
                i += 1;
                j += 1;
                h.update(arr, i);
            }
        }

        longest_match = find_longest_match(
            lmc,
            h,
            arr,
            i,
            inend,
            instart,
            ZOPFLI_MAX_MATCH,
            &mut Some(sublen.as_mut_slice()),
        );
        leng = longest_match.length;

        // Literal.
        if i < inend {
            let new_cost = costmodel(arr[i] as usize, 0) + f64::from(costs[j]);
            debug_assert!(new_cost >= 0.0);
            if new_cost < f64::from(costs[j + 1]) {
                costs[j + 1] = new_cost as f32;
                length_array[j + 1] = 1;
                dist_array[j + 1] = 0;
            }
        }
        // Lengths.
        let kend = cmp::min(leng as usize, inend - i);
        let mincostaddcostj = mincost + f64::from(costs[j]);

        for (k, &sublength) in sublen.iter().enumerate().take(kend + 1).skip(3) {
            // Calling the cost model is expensive, avoid this if we are already at
            // the minimum possible cost that it can return.
            if f64::from(costs[j + k]) <= mincostaddcostj {
                continue;
            }

            let new_cost = costmodel(k, sublength) + f64::from(costs[j]);
            debug_assert!(new_cost >= 0.0);
            if new_cost < f64::from(costs[j + k]) {
                debug_assert!(k <= ZOPFLI_MAX_MATCH);
                costs[j + k] = new_cost as f32;
                length_array[j + k] = k as u16;
                dist_array[j + k] = sublength;
            }
        }
        i += 1;
    }

    debug_assert!(costs[blocksize] >= 0.0);
    f64::from(costs[blocksize])
}

/// Calculates the optimal path of lz77 lengths to use, from the calculated
/// `length_array` and `dist_array`. Returns (length, dist) pairs in reverse order
/// (from end to start).
fn trace(size: usize, length_array: &[u16], dist_array: &[u16]) -> Vec<(u16, u16)> {
    let mut index = size;
    if size == 0 {
        return vec![];
    }
    let mut path = Vec::with_capacity(index);

    while index > 0 {
        let lai = length_array[index];
        let dai = dist_array[index];
        let laiu = lai as usize;
        path.push((lai, dai));
        debug_assert!(laiu <= index);
        debug_assert!(laiu <= ZOPFLI_MAX_MATCH);
        debug_assert_ne!(lai, 0);
        index -= laiu;
    }

    path
}

/// Does a single run for `lz77_optimal`. For good compression, repeated runs
/// with updated statistics should be performed.
/// `s`: the block state
/// `in_data`: the input data array
/// `instart`: where to start
/// `inend`: where to stop (not inclusive)
/// `length_array`: array of size `(inend - instart)` used to store lengths
/// `costmodel`: function to use as the cost model for this squeeze run
/// `store`: place to output the LZ77 data
/// returns the cost that was, according to the `costmodel`, needed to get to the end.
///     This is not the actual cost.
#[allow(clippy::too_many_arguments)] // Not feasible to refactor in a more readable way
fn lz77_optimal_run<F: Fn(usize, u16) -> f64, C: Cache>(
    lmc: &mut C,
    in_data: &[u8],
    instart: usize,
    inend: usize,
    costmodel: F,
    store: &mut Lz77Store,
    h: &mut ZopfliHash,
    costs: &mut Vec<f32>,
    length_array: &mut Vec<u16>,
    dist_array: &mut Vec<u16>,
    sublen: &mut Vec<u16>,
) {
    let cost = get_best_lengths(
        lmc,
        in_data,
        instart,
        inend,
        costmodel,
        h,
        costs,
        length_array,
        dist_array,
        sublen,
    );
    let path = trace(inend - instart, length_array, dist_array);
    store.store_from_path(in_data, instart, path);
    debug_assert!(cost < f64::INFINITY);
}

/// Does the same as `lz77_optimal`, but optimized for the fixed tree of the
/// deflate standard.
/// The fixed tree never gives the best compression. But this gives the best
/// possible LZ77 encoding possible with the fixed tree.
/// This does not create or output any fixed tree, only LZ77 data optimized for
/// using with a fixed tree.
/// If `instart` is larger than `0`, it uses values before `instart` as starting
/// dictionary.
pub fn lz77_optimal_fixed<C: Cache>(
    lmc: &mut C,
    in_data: &[u8],
    instart: usize,
    inend: usize,
    store: &mut Lz77Store,
) {
    let mut costs = Vec::with_capacity(inend - instart);
    let mut length_array = Vec::new();
    let mut dist_array = Vec::new();
    let mut sublen = Vec::new();
    lz77_optimal_run(
        lmc,
        in_data,
        instart,
        inend,
        get_cost_fixed,
        store,
        &mut ZopfliHash::new(),
        &mut costs,
        &mut length_array,
        &mut dist_array,
        &mut sublen,
    );
}

/// Calculates lit/len and dist pairs for given data.
/// If `instart` is larger than 0, it uses values before `instart` as starting
/// dictionary.
pub fn lz77_optimal<C: Cache>(
    lmc: &mut C,
    in_data: &[u8],
    instart: usize,
    inend: usize,
    max_iterations: u64,
    max_iterations_without_improvement: u64,
) -> Lz77Store {
    /* Dist to get to here with smallest cost. */
    let mut currentstore = Lz77Store::new();
    let mut outputstore = currentstore.clone();

    /* Initial run. */
    currentstore.greedy(lmc, in_data, instart, inend);
    let mut stats = SymbolStats::default();
    stats.get_statistics(&currentstore);

    let mut h = ZopfliHash::new();
    let mut costs = Vec::with_capacity(inend - instart + 1);
    let mut length_array = Vec::new();
    let mut dist_array = Vec::new();
    let mut sublen = Vec::new();

    let mut beststats = SymbolStats::default();

    let mut bestcost = f64::INFINITY;
    let mut lastcost = 0.0;
    /* Try randomizing the costs a bit once the size stabilizes. */
    let mut ran_state = RanState::new();
    let mut lastrandomstep = u64::MAX;

    /* Do regular deflate, then loop multiple shortest path runs, each time using
    the statistics of the previous run. */
    /* Repeat statistics with each time the cost model from the previous stat
    run. */
    let mut current_iteration: u64 = 0;
    let mut iterations_without_improvement: u64 = 0;
    loop {
        currentstore.reset();
        let cost_model = CostModel::from_stats(&stats);
        lz77_optimal_run(
            lmc,
            in_data,
            instart,
            inend,
            |a, b| cost_model.cost(a, b),
            &mut currentstore,
            &mut h,
            &mut costs,
            &mut length_array,
            &mut dist_array,
            &mut sublen,
        );
        let cost = calculate_block_size(&currentstore, 0, currentstore.size(), BlockType::Dynamic);

        if cost < bestcost {
            iterations_without_improvement = 0;
            /* Copy to the output store. */
            outputstore = currentstore.clone();
            beststats = stats;
            bestcost = cost;

            debug!("Iteration {current_iteration}: {cost} bit");
        } else {
            iterations_without_improvement += 1;
            trace!("Iteration {current_iteration}: {cost} bit");
            if iterations_without_improvement >= max_iterations_without_improvement {
                break;
            }
        }
        current_iteration += 1;
        if current_iteration >= max_iterations {
            break;
        }
        let laststats = stats;
        stats.clear_freqs();
        stats.get_statistics(&currentstore);
        if lastrandomstep != u64::MAX {
            /* This makes it converge slower but better. Do it only once the
            randomness kicks in so that if the user does few iterations, it gives a
            better result sooner. */
            stats = add_weighed_stat_freqs(&stats, 1.0, &laststats, 0.5);
            stats.calculate_entropy();
        }
        if current_iteration > 5 && (cost - lastcost).abs() < f64::EPSILON {
            stats = beststats;
            stats.randomize_stat_freqs(&mut ran_state);
            stats.calculate_entropy();
            lastrandomstep = current_iteration;
        }
        lastcost = cost;
    }
    outputstore
}
