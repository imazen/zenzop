#![forbid(unsafe_code)]
#![deny(trivial_casts, trivial_numeric_casts, missing_docs)]

//! A faster fork of the [Zopfli](https://github.com/google/zopfli) compression library.
//!
//! Zopfli is a DEFLATE compressor that produces near-optimal output at the cost of speed.
//! zenzop produces identical output to zopfli but runs 1.2–2x faster through algorithmic
//! improvements: precomputed cost tables, SIMD-accelerated match comparison, and a skip-hash
//! optimization that eliminates redundant hash chain walks on cached iterations.
//!
//! With [`Options::enhanced`] enabled, zenzop applies ECT-derived optimizations — expanded
//! precode search, multi-strategy Huffman tree selection, and enhanced parser diversification —
//! to produce smaller output than standard Zopfli.
//!
//! # Features
//!
//! This crate exposes the following features. You can enable or disable them in your `Cargo.toml`
//! as needed.
//!
//! - `gzip` (enabled by default): enables support for compression in the gzip format.
//! - `zlib` (enabled by default): enables support for compression in the Zlib format.
//! - `std` (enabled by default): enables linking against the Rust standard library. When not enabled,
//!   the crate is built with the `#![no_std]` attribute and can be used in any environment where
//!   [`alloc`](https://doc.rust-lang.org/alloc/) (i.e., a memory allocator) is available. In addition,
//!   the crate exposes minimalist versions of the `std` I/O traits it needs to function, allowing users
//!   to implement them.

#![cfg_attr(not(feature = "std"), no_std)]

// No-op log implementation for no-std targets
#[cfg(not(feature = "std"))]
macro_rules! debug {
    ( $( $_:expr ),* ) => {};
}
#[cfg(not(feature = "std"))]
macro_rules! trace {
    ( $( $_:expr ),* ) => {};
}
#[cfg(not(feature = "std"))]
macro_rules! log_enabled {
    ( $( $_:expr ),* ) => {
        false
    };
}

#[cfg_attr(not(feature = "std"), macro_use)]
extern crate alloc;

pub use deflate::{BlockType, CompressResult, DeflateEncoder};
pub use enough::{Stop, StopReason, Unstoppable};
#[cfg(feature = "gzip")]
pub use gzip::GzipEncoder;
#[cfg(all(test, feature = "std"))]
use proptest::prelude::*;
#[cfg(feature = "zlib")]
pub use zlib::ZlibEncoder;

mod blocksplitter;
mod cache;
mod deflate;
#[cfg(feature = "gzip")]
mod gzip;
mod hash;
#[cfg(any(doc, not(feature = "std")))]
mod io;
mod iter;
mod katajainen;
mod lz77;
#[cfg(not(feature = "std"))]
mod math;
mod squeeze;
mod symbols;
mod tree;
mod util;
#[cfg(feature = "zlib")]
mod zlib;

use core::num::NonZeroU64;
#[cfg(all(not(doc), feature = "std"))]
use std::io::{Error, Write};

#[cfg(any(doc, not(feature = "std")))]
pub use io::{Error, ErrorKind, Write};

/// Convert a [`StopReason`] into the crate's I/O error type.
#[cfg(all(not(doc), feature = "std"))]
fn stop_to_error(reason: StopReason) -> Error {
    Error::other(match reason {
        StopReason::Cancelled => "operation cancelled",
        StopReason::TimedOut => "operation timed out",
        _ => "operation stopped",
    })
}

/// Convert a [`StopReason`] into the crate's I/O error type.
#[cfg(any(doc, not(feature = "std")))]
fn stop_to_error(_reason: StopReason) -> Error {
    io::ErrorKind::Cancelled.into()
}

/// Maximum number of squeeze iterations the encoder will actually run, regardless of
/// what [`Options::iteration_count`] or [`Options::iterations_without_improvement`] is
/// set to.
///
/// Each squeeze iteration runs a full DP pass over the block, so wall-clock cost grows
/// linearly with iteration count and the marginal compression gain is well past
/// negligible by the low hundreds. Real-world good values are 5–15 (single-digit ms to
/// seconds per MiB depending on input); anything beyond ~1000 is pathological and only
/// burns CPU. Capping internally avoids a compute-DoS footgun for callers that surface
/// `Options` to untrusted config (e.g., a server endpoint that takes
/// `iteration_count` as a query parameter).
///
/// Callers that want a higher cap can call [`Options::with_iteration_cap`].
pub const DEFAULT_MAX_ITERATIONS: u64 = 1000;

/// Options for the Zopfli compression algorithm.
///
/// # Examples
///
/// ```
/// use zenzop::Options;
/// use core::num::NonZeroU64;
///
/// // Use defaults (15 iterations)
/// let opts = Options::default();
/// assert_eq!(opts.iteration_count.get(), 15);
///
/// // Customize iteration count for faster compression
/// let mut opts = Options::default();
/// opts.iteration_count = NonZeroU64::new(5).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(all(test, feature = "std"), derive(proptest_derive::Arbitrary))]
#[non_exhaustive]
pub struct Options {
    /// Maximum amount of times to rerun forward and backward pass to optimize LZ77
    /// compression cost.
    /// Good values: 10, 15 for small files, 5 for files over several MB in size or
    /// it will be too slow.
    ///
    /// **The encoder internally clamps the effective iteration count to
    /// [`Options::iteration_cap`] (default [`DEFAULT_MAX_ITERATIONS`] = 1000) to prevent
    /// pathological compute-DoS configurations (e.g. `u64::MAX`).** Use
    /// [`Options::with_iteration_cap`] to raise the cap if you genuinely need more
    /// iterations.
    ///
    /// Default value: 15.
    #[cfg_attr(
        all(test, feature = "std"),
        proptest(
            strategy = "(1..=10u64).prop_map(|iteration_count| NonZeroU64::new(iteration_count).unwrap())"
        )
    )]
    pub iteration_count: NonZeroU64,
    /// Stop after rerunning forward and backward pass this many times without finding
    /// a smaller representation of the block.
    ///
    /// **Also clamped internally by [`Options::iteration_cap`].** With
    /// `iterations_without_improvement = u64::MAX` and no [`Stop`] wired up, the encoder
    /// would otherwise loop until heat death; the cap turns that into a finite worst case.
    ///
    /// Default value: practically infinite (maximum `u64` value), clamped at runtime.
    pub iterations_without_improvement: NonZeroU64,
    /// Maximum amount of blocks to split into (0 for unlimited, but this can give
    /// extreme results that hurt compression on some files).
    ///
    /// Default value: 15.
    pub maximum_block_splits: u16,
    /// The type of DEFLATE blocks to generate.
    ///
    /// [`Dynamic`](BlockType::Dynamic) (the default) lets the algorithm choose the
    /// most space-efficient block types automatically. Use other variants only for
    /// testing or specialized scenarios.
    ///
    /// Default value: [`BlockType::Dynamic`].
    pub block_type: BlockType,
    /// Enable ECT-derived optimizations for smaller output.
    ///
    /// When `true`, applies expanded precode search (A1), multi-strategy
    /// Huffman tree selection (A2), and enhanced parser diversification (A3).
    /// This produces smaller DEFLATE output at the cost of byte-for-byte
    /// parity with upstream C Zopfli.
    ///
    /// Default value: `false`.
    pub enhanced: bool,
    /// Internal cap on the effective iteration count.
    ///
    /// Both [`Options::iteration_count`] and [`Options::iterations_without_improvement`]
    /// are silently clamped to this value before each block is encoded. This prevents
    /// compute-DoS from a `u64::MAX`-style configuration while leaving real-world good
    /// values (5–15) untouched.
    ///
    /// Use [`Options::with_iteration_cap`] to raise this if you genuinely need more
    /// iterations than the default. Setting this lower than `iteration_count` will
    /// effectively reduce iteration count for that encode.
    ///
    /// Default value: [`DEFAULT_MAX_ITERATIONS`] (1000).
    #[cfg_attr(
        all(test, feature = "std"),
        proptest(
            strategy = "(1..=DEFAULT_MAX_ITERATIONS).prop_map(|n| NonZeroU64::new(n).unwrap())"
        )
    )]
    pub iteration_cap: NonZeroU64,
}

impl Options {
    /// Set the internal cap on effective iteration count.
    ///
    /// See [`Options::iteration_cap`] for what this controls.
    pub fn with_iteration_cap(mut self, cap: NonZeroU64) -> Self {
        self.iteration_cap = cap;
        self
    }

    /// The clamped iteration count actually used by the encoder.
    ///
    /// Equal to `min(iteration_count, iteration_cap)`.
    pub fn effective_iteration_count(&self) -> u64 {
        self.iteration_count.get().min(self.iteration_cap.get())
    }

    /// The clamped "iterations without improvement" budget actually used by the encoder.
    ///
    /// Equal to `min(iterations_without_improvement, iteration_cap)`.
    pub fn effective_iterations_without_improvement(&self) -> u64 {
        self.iterations_without_improvement
            .get()
            .min(self.iteration_cap.get())
    }
}

impl Default for Options {
    fn default() -> Self {
        Self {
            iteration_count: NonZeroU64::new(15).unwrap(),
            iterations_without_improvement: NonZeroU64::new(u64::MAX).unwrap(),
            maximum_block_splits: 15,
            block_type: BlockType::Dynamic,
            enhanced: false,
            iteration_cap: NonZeroU64::new(DEFAULT_MAX_ITERATIONS).unwrap(),
        }
    }
}

/// The output file format to use to store data compressed with Zopfli.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[cfg(feature = "std")]
pub enum Format {
    /// The gzip file format, as defined in
    /// [RFC 1952](https://datatracker.ietf.org/doc/html/rfc1952).
    ///
    /// This file format can be easily decompressed with the gzip
    /// program.
    #[cfg(feature = "gzip")]
    Gzip,
    /// The zlib file format, as defined in
    /// [RFC 1950](https://datatracker.ietf.org/doc/html/rfc1950).
    ///
    /// The zlib format has less header overhead than gzip, but it
    /// stores less metadata.
    #[cfg(feature = "zlib")]
    Zlib,
    /// The raw DEFLATE stream format, as defined in
    /// [RFC 1951](https://datatracker.ietf.org/doc/html/rfc1951).
    ///
    /// Raw DEFLATE streams are not meant to be stored as-is because
    /// they lack error detection and correction metadata. They
    /// are usually embedded in other file formats, such as gzip
    /// and zlib.
    Deflate,
}

#[cfg(feature = "std")]
fn feed_master_blocks<R: std::io::Read, E: Write>(
    mut in_data: R,
    encoder: &mut E,
) -> Result<(), Error> {
    let mut buf = vec![0u8; crate::util::ZOPFLI_MASTER_BLOCK_SIZE];
    loop {
        let mut filled = 0;
        while filled < buf.len() {
            match in_data.read(&mut buf[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        if filled == 0 {
            break;
        }
        encoder.write_all(&buf[..filled])?;
        if filled < buf.len() {
            break;
        }
    }
    Ok(())
}

/// Compresses data from a source with the Zopfli algorithm, using the specified
/// options, and writes the result to a sink in the defined output format.
///
/// # Examples
///
/// ```
/// use zenzop::{Options, Format};
///
/// let data = b"The quick brown fox jumps over the lazy dog";
/// let mut compressed = Vec::new();
/// zenzop::compress(Options::default(), Format::Deflate, &data[..], &mut compressed).unwrap();
/// assert!(!compressed.is_empty());
/// ```
#[cfg(feature = "std")]
pub fn compress<R: std::io::Read, W: Write>(
    options: Options,
    output_format: Format,
    in_data: R,
    out: W,
) -> Result<(), Error> {
    match output_format {
        #[cfg(feature = "gzip")]
        Format::Gzip => {
            let mut encoder = GzipEncoder::new(options, out)?;
            feed_master_blocks(in_data, &mut encoder)?;
            encoder.finish().map(|_| ())
        }
        #[cfg(feature = "zlib")]
        Format::Zlib => {
            let mut encoder = ZlibEncoder::new(options, out)?;
            feed_master_blocks(in_data, &mut encoder)?;
            encoder.finish().map(|_| ())
        }
        Format::Deflate => {
            let mut encoder = DeflateEncoder::new(options, out);
            feed_master_blocks(in_data, &mut encoder)?;
            encoder.finish().map(|_| ())
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod test {
    use std::io;

    use miniz_oxide::inflate;
    use proptest::proptest;

    use super::*;

    proptest! {
        #[test]
        fn deflating_is_reversible(
            options: Options,
            data in prop::collection::vec(any::<u8>(), 0..64 * 1024)
        ) {
            let mut compressed_data = Vec::with_capacity(data.len());

            let mut encoder = DeflateEncoder::new(options, &mut compressed_data);
            io::copy(&mut &*data, &mut encoder).unwrap();
            encoder.finish().unwrap();

            let decompressed_data = inflate::decompress_to_vec(&compressed_data).expect("Could not inflate compressed stream");
            prop_assert_eq!(data, decompressed_data, "Decompressed data should match input data");
        }
    }

    #[test]
    fn enhanced_produces_valid_deflate() {
        let data = b"The quick brown fox jumps over the lazy dog. \
                     The quick brown fox jumps over the lazy dog. \
                     The quick brown fox jumps over the lazy dog.";
        let mut compressed = Vec::new();
        let options = Options {
            enhanced: true,
            iteration_count: core::num::NonZeroU64::new(5).unwrap(),
            ..Options::default()
        };
        let mut encoder = DeflateEncoder::new(options, &mut compressed);
        io::copy(&mut &data[..], &mut encoder).unwrap();
        encoder.finish().unwrap();

        let decompressed =
            inflate::decompress_to_vec(&compressed).expect("Enhanced DEFLATE output must be valid");
        assert_eq!(&data[..], &decompressed[..]);
    }

    #[test]
    fn enhanced_output_no_larger_than_standard() {
        // Generate test data with enough structure for compression to matter
        let mut data = Vec::with_capacity(16384);
        for i in 0..16384u16 {
            // Mix of repetitive and varied content
            data.push((i % 256) as u8);
            if i % 7 == 0 {
                data.extend_from_slice(b"repeat");
            }
        }

        let compress = |enhanced: bool| -> Vec<u8> {
            let mut compressed = Vec::new();
            let options = Options {
                enhanced,
                iteration_count: core::num::NonZeroU64::new(10).unwrap(),
                ..Options::default()
            };
            let mut encoder = DeflateEncoder::new(options, &mut compressed);
            io::copy(&mut &data[..], &mut encoder).unwrap();
            encoder.finish().unwrap();
            compressed
        };

        let standard = compress(false);
        let enhanced = compress(true);

        assert!(
            enhanced.len() <= standard.len(),
            "Enhanced ({} bytes) should be <= standard ({} bytes)",
            enhanced.len(),
            standard.len()
        );

        // Verify enhanced output decompresses correctly
        let decompressed =
            inflate::decompress_to_vec(&enhanced).expect("Enhanced output must decompress");
        assert_eq!(&data[..], &decompressed[..]);
    }

    /// Extract filtered IDAT data from a PNG file by decompressing the zlib stream.
    fn extract_png_idat(png_data: &[u8]) -> Vec<u8> {
        // Parse PNG chunks, concatenate IDAT data
        let mut idat_data = Vec::new();
        let mut pos = 8; // Skip PNG signature
        while pos + 12 <= png_data.len() {
            let len = u32::from_be_bytes(png_data[pos..pos + 4].try_into().unwrap()) as usize;
            let chunk_type = &png_data[pos + 4..pos + 8];
            if chunk_type == b"IDAT" {
                idat_data.extend_from_slice(&png_data[pos + 8..pos + 8 + len]);
            }
            pos += 12 + len; // length + type + data + CRC
        }
        // Decompress the zlib stream to get filtered scanline data
        inflate::decompress_to_vec_zlib(&idat_data).expect("Failed to decompress IDAT")
    }

    #[test]
    fn enhanced_vs_standard_on_png_idat() {
        let test_pngs: &[(&str, &[u8])] = &[
            ("eeyore.png", include_bytes!("../test/data/eeyore.png")),
            (
                "heartbleed.png",
                include_bytes!("../test/data/heartbleed.png"),
            ),
            ("computer.png", include_bytes!("../test/data/computer.png")),
        ];

        for &(name, png_data) in test_pngs {
            let filtered = extract_png_idat(png_data);

            let compress = |enhanced: bool| -> Vec<u8> {
                let mut compressed = Vec::new();
                let options = Options {
                    enhanced,
                    iteration_count: core::num::NonZeroU64::new(15).unwrap(),
                    ..Options::default()
                };
                let mut encoder = DeflateEncoder::new(options, &mut compressed);
                io::copy(&mut &*filtered, &mut encoder).unwrap();
                encoder.finish().unwrap();
                compressed
            };

            let standard = compress(false);
            let enhanced = compress(true);

            eprintln!(
                "{name}: idat={} bytes, standard={} enhanced={} saved={} ({:.3}%)",
                filtered.len(),
                standard.len(),
                enhanced.len(),
                standard.len() as i64 - enhanced.len() as i64,
                (1.0 - enhanced.len() as f64 / standard.len() as f64) * 100.0,
            );

            let decompressed =
                inflate::decompress_to_vec(&enhanced).expect("Enhanced output must decompress");
            assert_eq!(filtered, decompressed);
        }
    }

    #[test]
    fn enhanced_vs_standard_size_report() {
        let test_files: &[(&str, &[u8])] = &[
            (
                "calgary-books-32k",
                &include_bytes!("../test/data/calgary-books.txt")[..32768],
            ),
            (
                "codetriage.js",
                include_bytes!("../test/data/codetriage.js"),
            ),
        ];

        for &(name, data) in test_files {
            let compress = |enhanced: bool| -> Vec<u8> {
                let mut compressed = Vec::new();
                let options = Options {
                    enhanced,
                    iteration_count: core::num::NonZeroU64::new(15).unwrap(),
                    ..Options::default()
                };
                let mut encoder = DeflateEncoder::new(options, &mut compressed);
                io::copy(&mut &*data, &mut encoder).unwrap();
                encoder.finish().unwrap();
                compressed
            };

            let standard = compress(false);
            let enhanced = compress(true);

            eprintln!(
                "{name} ({} bytes): standard={} enhanced={} saved={} ({:.3}%)",
                data.len(),
                standard.len(),
                enhanced.len(),
                standard.len() as i64 - enhanced.len() as i64,
                (1.0 - enhanced.len() as f64 / standard.len() as f64) * 100.0,
            );

            let decompressed =
                inflate::decompress_to_vec(&enhanced).expect("Enhanced output must decompress");
            assert_eq!(data, &decompressed[..]);
        }
    }

    /// Default options must clamp at `DEFAULT_MAX_ITERATIONS`.
    #[test]
    fn iteration_cap_default_value() {
        let opts = Options::default();
        assert_eq!(opts.iteration_cap.get(), DEFAULT_MAX_ITERATIONS);
    }

    /// `effective_iteration_count` must clamp pathological values.
    #[test]
    fn iteration_cap_clamps_huge_count() {
        let opts = Options {
            iteration_count: core::num::NonZeroU64::new(u64::MAX).unwrap(),
            ..Options::default()
        };
        assert_eq!(opts.effective_iteration_count(), DEFAULT_MAX_ITERATIONS);
    }

    /// `effective_iteration_count` must NOT clamp normal values.
    #[test]
    fn iteration_cap_passes_normal_count() {
        let opts = Options {
            iteration_count: core::num::NonZeroU64::new(15).unwrap(),
            ..Options::default()
        };
        assert_eq!(opts.effective_iteration_count(), 15);
    }

    /// Default `iterations_without_improvement = u64::MAX` must clamp.
    #[test]
    fn iteration_cap_clamps_default_no_improvement() {
        let opts = Options::default();
        assert_eq!(
            opts.effective_iterations_without_improvement(),
            DEFAULT_MAX_ITERATIONS
        );
    }

    /// `with_iteration_cap` must allow raising the cap.
    #[test]
    fn iteration_cap_can_be_raised() {
        let raised = NonZeroU64::new(50_000).unwrap();
        let opts = Options::default().with_iteration_cap(raised);
        assert_eq!(opts.iteration_cap, raised);
        // Now `iterations_without_improvement = u64::MAX` still gets clamped, but to the
        // new cap, not the old default.
        assert_eq!(
            opts.effective_iterations_without_improvement(),
            raised.get()
        );
    }

    /// End-to-end: compressing with `iteration_count = u64::MAX` must terminate quickly
    /// and produce valid DEFLATE output. Without the cap this would loop until heat death.
    #[test]
    fn iteration_cap_enables_pathological_options_to_terminate() {
        let data = b"The quick brown fox jumps over the lazy dog. \
                     The quick brown fox jumps over the lazy dog. \
                     The quick brown fox jumps over the lazy dog.";
        let mut compressed = Vec::new();
        let options = Options {
            iteration_count: NonZeroU64::new(u64::MAX).unwrap(),
            iterations_without_improvement: NonZeroU64::new(u64::MAX).unwrap(),
            // Lower the cap to keep this test fast — proves the cap path actually
            // takes effect rather than just relying on the 1000 default.
            iteration_cap: NonZeroU64::new(2).unwrap(),
            ..Options::default()
        };
        let mut encoder = DeflateEncoder::new(options, &mut compressed);
        io::copy(&mut &data[..], &mut encoder).unwrap();
        encoder.finish().unwrap();

        let decompressed = inflate::decompress_to_vec(&compressed)
            .expect("Output of capped pathological options must be valid DEFLATE");
        assert_eq!(&data[..], &decompressed[..]);
    }

    struct ChunkedReader<R> {
        inner: R,
        chunk_size: usize,
        interrupt_every: usize,
        calls: usize,
    }

    impl<R> ChunkedReader<R> {
        const fn new(inner: R, chunk_size: usize, interrupt_every: usize) -> Self {
            Self {
                inner,
                chunk_size,
                interrupt_every,
                calls: 0,
            }
        }
    }

    impl<R: io::Read> io::Read for ChunkedReader<R> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.calls += 1;
            if self.interrupt_every > 0 && self.calls.is_multiple_of(self.interrupt_every) {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "simulated EINTR",
                ));
            }
            let limit = buf.len().min(self.chunk_size);
            self.inner.read(&mut buf[..limit])
        }
    }

    #[test]
    fn compress_output_independent_of_read_chunking() {
        // > 1.2 MiB (1_258_291 B) of compressible structured data crossing
        // ZOPFLI_MASTER_BLOCK_SIZE (1_000_000). Varied runs let Zopfli's
        // Hash2/RLE match path finish quickly in unoptimized debug builds
        // while still exercising dynamic Huffman trees and block splitting
        // across the 1 MB master-block boundary.
        const LEN: usize = 1_260_000;
        let mut data = Vec::with_capacity(LEN);
        let mut block_idx: u32 = 0;
        while data.len() < LEN {
            let mut state = block_idx
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add(0x1234_5678);
            for _ in 0..16 {
                if data.len() == LEN {
                    break;
                }
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                data.push((state >> 24) as u8);
            }
            // Run >= 4 * ZOPFLI_MAX_MATCH lets squeeze's RLE shortcut skip
            // full DP walks in debug builds while varied run bytes keep
            // block splitting and Huffman trees sensitive to master blocks.
            let run_byte = (block_idx % 251) as u8;
            let run_len = (LEN - data.len()).min(4080);
            data.resize(data.len() + run_len, run_byte);
            block_idx += 1;
        }

        let options = Options {
            iteration_count: NonZeroU64::new(1).unwrap(),
            ..Options::default()
        };

        let formats: &[Format] = &[
            #[cfg(feature = "gzip")]
            Format::Gzip,
            #[cfg(feature = "zlib")]
            Format::Zlib,
            Format::Deflate,
        ];

        for &format in formats {
            let mut ref_slice = Vec::new();
            compress(options, format, data.as_slice(), &mut ref_slice).unwrap();
            assert!(
                ref_slice.len() < data.len() / 2,
                "expected compressible test fixture, got {} / {} bytes",
                ref_slice.len(),
                data.len()
            );

            let mut ref_cursor = Vec::new();
            compress(options, format, io::Cursor::new(&data), &mut ref_cursor).unwrap();
            assert_eq!(
                ref_slice, ref_cursor,
                "slice and Cursor differed for {format:?}"
            );

            for &(chunk_size, interrupt_every) in
                &[(7, 0), (4096, 0), (4096, 29), (8192, 0), (131_071, 0)]
            {
                let reader = ChunkedReader::new(data.as_slice(), chunk_size, interrupt_every);
                let mut chunked = Vec::new();
                compress(options, format, reader, &mut chunked).unwrap();
                assert_eq!(
                    ref_slice, chunked,
                    "chunk_size={chunk_size}, interrupt_every={interrupt_every} differed for {format:?}"
                );
            }

            assert_roundtrip(format, &ref_slice, &data);
        }
    }

    fn assert_roundtrip(format: Format, compressed: &[u8], expected: &[u8]) {
        match format {
            #[cfg(feature = "gzip")]
            Format::Gzip => {
                let mut decompressed = vec![0u8; expected.len()];
                let outcome = zenflate::Decompressor::new()
                    .gzip_decompress(compressed, &mut decompressed, Unstoppable)
                    .unwrap();
                assert_eq!(&decompressed[..outcome.output_written], expected);
            }
            #[cfg(feature = "zlib")]
            Format::Zlib => {
                let decompressed = inflate::decompress_to_vec_zlib(compressed).unwrap();
                assert_eq!(decompressed, expected);
            }
            Format::Deflate => {
                let decompressed = inflate::decompress_to_vec(compressed).unwrap();
                assert_eq!(decompressed, expected);
            }
        }
    }

    #[test]
    fn compress_empty_and_exact_master_block_boundary() {
        let options = Options {
            iteration_count: NonZeroU64::new(1).unwrap(),
            ..Options::default()
        };
        let formats: &[Format] = &[
            #[cfg(feature = "gzip")]
            Format::Gzip,
            #[cfg(feature = "zlib")]
            Format::Zlib,
            Format::Deflate,
        ];

        // 0 bytes, exact ZOPFLI_MASTER_BLOCK_SIZE (1_000_000), and 1_000_001 bytes.
        let mut boundary_data = vec![b'A'; 500_000];
        boundary_data.resize(crate::util::ZOPFLI_MASTER_BLOCK_SIZE, b'B');
        boundary_data.push(b'C');

        for &len in &[
            0usize,
            crate::util::ZOPFLI_MASTER_BLOCK_SIZE,
            crate::util::ZOPFLI_MASTER_BLOCK_SIZE + 1,
        ] {
            let input = &boundary_data[..len];
            for &format in formats {
                let mut ref_out = Vec::new();
                compress(options, format, input, &mut ref_out).unwrap();

                let mut chunked_out = Vec::new();
                let reader = ChunkedReader::new(input, 131_071, 3);
                compress(options, format, reader, &mut chunked_out).unwrap();

                assert_eq!(
                    ref_out, chunked_out,
                    "boundary len={len} differed for {format:?}"
                );
                assert_roundtrip(format, &ref_out, input);
            }
        }
    }
}
