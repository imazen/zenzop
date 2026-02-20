use alloc::boxed::Box;
use core::cmp;

use crate::util::{ZOPFLI_MIN_MATCH, ZOPFLI_WINDOW_MASK, ZOPFLI_WINDOW_SIZE};

const HASH_SHIFT: i32 = 5;
const HASH_MASK: u16 = 32767;
/// Sentinel value indicating "no entry" — valid hash values are 0..32767.
const HASH_NONE: u16 = u16::MAX;

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Which {
    Hash1,
    Hash2,
}

#[derive(Clone)]
struct HashThing {
    head: Box<[u16; 65536]>, /* Hash value to index of its most recent occurrence. */
    prev: Box<[u16; ZOPFLI_WINDOW_SIZE]>, /* Index to prev occurrence of same hash. */
    hashval: Box<[u16; ZOPFLI_WINDOW_SIZE]>, /* Hash value at this index. HASH_NONE = unset. */
    val: u16,                /* Current hash value. */
}

impl HashThing {
    fn new() -> Self {
        let mut prev: Box<[u16; ZOPFLI_WINDOW_SIZE]> = vec![0u16; ZOPFLI_WINDOW_SIZE]
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| unreachable!());
        for (i, p) in prev.iter_mut().enumerate() {
            *p = i as u16;
        }
        Self {
            head: vec![HASH_NONE; 65536]
                .into_boxed_slice()
                .try_into()
                .unwrap_or_else(|_| unreachable!()),
            prev,
            hashval: vec![HASH_NONE; ZOPFLI_WINDOW_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap_or_else(|_| unreachable!()),
            val: 0,
        }
    }

    fn reset(&mut self) {
        self.head.fill(HASH_NONE);
        for (i, p) in self.prev.iter_mut().enumerate() {
            *p = i as u16;
        }
        self.hashval.fill(HASH_NONE);
        self.val = 0;
    }

    fn update(&mut self, hpos: usize) {
        let hashval = self.val;
        let index = self.val as usize;
        let head_index = self.head[index];
        let prev = if head_index != HASH_NONE && self.hashval[head_index as usize] == self.val {
            head_index
        } else {
            hpos as u16
        };

        self.prev[hpos] = prev;
        self.hashval[hpos] = hashval;
        self.head[index] = hpos as u16;
    }
}

#[derive(Clone)]
pub struct ZopfliHash {
    hash1: HashThing,
    hash2: HashThing,
    pub same: Box<[u16; ZOPFLI_WINDOW_SIZE]>, /* Amount of repetitions of same byte after this. */
}

impl ZopfliHash {
    pub fn new() -> Box<Self> {
        Box::new(Self {
            hash1: HashThing::new(),
            hash2: HashThing::new(),
            same: vec![0u16; ZOPFLI_WINDOW_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap_or_else(|_| unreachable!()),
        })
    }

    pub fn reset(&mut self) {
        self.hash1.reset();
        self.hash2.reset();
        self.same.fill(0);
    }

    pub fn warmup(&mut self, arr: &[u8], pos: usize, end: usize) {
        let c = arr[pos];
        self.update_val(c);

        if pos + 1 < end {
            let c = arr[pos + 1];
            self.update_val(c);
        }
    }

    /// Update the sliding hash value with the given byte. All calls to this function
    /// must be made on consecutive input characters. Since the hash value exists out
    /// of multiple input bytes, a few warmups with this function are needed initially.
    fn update_val(&mut self, c: u8) {
        self.hash1.val = ((self.hash1.val << HASH_SHIFT) ^ u16::from(c)) & HASH_MASK;
    }

    pub fn update(&mut self, array: &[u8], pos: usize) {
        let hash_value = array.get(pos + ZOPFLI_MIN_MATCH - 1).copied().unwrap_or(0);
        self.update_val(hash_value);

        let hpos = pos & ZOPFLI_WINDOW_MASK;

        self.hash1.update(hpos);

        // Update "same".
        let mut amount: u16 = 0;
        let same = self.same[pos.wrapping_sub(1) & ZOPFLI_WINDOW_MASK];
        if same > 1 {
            amount = same - 1;
        }

        let array_pos = array[pos];
        let start = pos + amount as usize + 1;
        let scan_end = cmp::min(pos + u16::MAX as usize + 1, array.len());
        if start < scan_end {
            for &byte in &array[start..scan_end] {
                if byte != array_pos {
                    break;
                }
                amount += 1;
            }
        }

        self.same[hpos] = amount;

        self.hash2.val = (amount.wrapping_sub(ZOPFLI_MIN_MATCH as u16) & 255) ^ self.hash1.val;

        self.hash2.update(hpos);
    }

    pub fn prev_at(&self, index: usize, which: Which) -> usize {
        // Mask ensures the returned value is always < ZOPFLI_WINDOW_SIZE,
        // letting LLVM eliminate downstream bounds checks on same[]/prev[]/hashval[].
        (match which {
            Which::Hash1 => self.hash1.prev[index],
            Which::Hash2 => self.hash2.prev[index],
        }) as usize
            & ZOPFLI_WINDOW_MASK
    }

    pub fn hash_val_at(&self, index: usize, which: Which) -> i32 {
        let hashval = match which {
            Which::Hash1 => self.hash1.hashval[index],
            Which::Hash2 => self.hash2.hashval[index],
        };
        if hashval == HASH_NONE {
            -1
        } else {
            hashval as i32
        }
    }

    pub fn val(&self, which: Which) -> u16 {
        match which {
            Which::Hash1 => self.hash1.val,
            Which::Hash2 => self.hash2.val,
        }
    }
}
