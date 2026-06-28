<!-- GENERATED FROM README.md by zenutils gen-readme-crates.sh — DO NOT EDIT. -->

# zenzop [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenzop/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zenzop/actions/workflows/ci.yml)

A faster fork of the [Zopfli](https://github.com/google/zopfli) DEFLATE compressor, written in Rust. Pure Rust, `#![forbid(unsafe_code)]`, `no_std` + `alloc` compatible.

Zopfli is a well-known, battle-tested DEFLATE compressor that produces near-optimal output at the cost of speed. zenzop produces byte-identical output 1.2–2x faster through algorithmic improvements: precomputed cost tables, SIMD-accelerated match comparison, arena-based Huffman tree construction, pre-allocated stores, and a skip-hash optimization that eliminates redundant hash-chain walks on cached iterations.

With `enhanced` mode enabled, zenzop applies optimizations derived from [ECT](https://github.com/fhanau/Efficient-Compression-Tool) — expanded precode search, multi-strategy Huffman tree selection, and enhanced parser diversification — to produce smaller output than standard Zopfli, trading away byte-for-byte parity with the C reference.

**zenzop is compress-only.** Like Zopfli itself, it has no decompressor — the output is standard gzip/zlib/raw DEFLATE and decodes with any conforming decoder (e.g. [`flate2`](https://crates.io/crates/flate2), [`miniz_oxide`](https://crates.io/crates/miniz_oxide), or [zenflate](https://crates.io/crates/zenflate)'s decode side).

## Quick start

```toml
[dependencies]
zenzop = "0.4.2"
```

```rust
use zenzop::{Options, Format};

let data = b"The quick brown fox jumps over the lazy dog";
let mut compressed = Vec::new();
zenzop::compress(Options::default(), Format::Gzip, &data[..], &mut compressed).unwrap();
assert!(!compressed.is_empty());
```

`compress` is the one-call entry point: pick a `Format` (`Gzip`, `Zlib`, or raw `Deflate`), pass any `Read` source and `Write` sink, and zenzop streams the optimized output. For finer control — enhanced mode, iteration count, cancellation — use the `Options` fields and streaming encoders documented below.

## Features

- **Byte-identical output** to C Zopfli in the default mode (verified by golden-master tests in CI).
- **Enhanced mode** for smaller output than standard Zopfli (`Options::enhanced`).
- **1.2–2x faster** than Zopfli (input-dependent; larger gains on smaller blocks). See [benchmarks/](https://github.com/imazen/zenzop/blob/main/benchmarks/README.md).
- **`no_std` + `alloc`** — works on embedded and WASM targets.
- **Cooperative cancellation** via [`enough::Stop`](https://docs.rs/enough) — cancel long-running compressions cleanly.
- **Streaming encoder API** — `DeflateEncoder`, `GzipEncoder`, `ZlibEncoder`.
- **Parallel block compression** with the optional `parallel` (rayon) feature.

## Enhanced mode

```rust
let mut options = zenzop::Options::default();
options.enhanced = true;

let mut output = Vec::new();
zenzop::compress(options, zenzop::Format::Gzip, &b"Hello, world!"[..], &mut output).unwrap();
```

Enhanced mode produces smaller DEFLATE output than standard Zopfli with roughly 5% runtime overhead. The output is still valid DEFLATE; it simply no longer matches the C reference byte-for-byte.

## Tuning the ratio (iteration count)

Like Zopfli, zenzop trades CPU time for ratio by re-running its forward/backward LZ77 optimization pass multiple times. The knob is **`Options::iteration_count`** (default 15) — more iterations means smaller output and more time:

```rust
use std::num::NonZeroU64;

// `Options` is `#[non_exhaustive]`, so build from `default()` and set fields.
let mut options = zenzop::Options::default();
options.enhanced = true;
options.iteration_count = NonZeroU64::new(60).unwrap();   // squeeze harder for maximum ratio

let mut output = Vec::new();
zenzop::compress(options, zenzop::Format::Gzip, &b"Hello, world!"[..], &mut output).unwrap();
```

The effective iteration count is internally clamped to `Options::iteration_cap` (default `DEFAULT_MAX_ITERATIONS` = 1000) so that an `Options` fed from untrusted config can't trigger a compute-DoS; 60 is well under the cap. If you genuinely need more than 1000 iterations, raise the cap with [`Options::with_iteration_cap`](https://docs.rs/zenzop/latest/zenzop/struct.Options.html#method.with_iteration_cap).

### `Options` fields

| Field | Type | Default | Effect |
|-------|------|---------|--------|
| `iteration_count` | `NonZeroU64` | `15` | Total LZ77 optimization passes. Higher = smaller output, slower. Raise (e.g. to `60`) for maximum ratio. |
| `enhanced` | `bool` | `false` | Enable ECT-derived optimizations (smaller output, ~5% slower, drops byte-for-byte parity with C Zopfli). |
| `iterations_without_improvement` | `NonZeroU64` | `u64::MAX` | Early-stop budget: bail after this many consecutive passes with no size improvement. Defaults to "never give up early". |
| `maximum_block_splits` | `u16` | `15` | Maximum number of block splits (`0` = unlimited). |
| `block_type` | `BlockType` | `BlockType::Dynamic` | DEFLATE block type; `Dynamic` auto-selects the smallest. |
| `iteration_cap` | `NonZeroU64` | `1000` | Internal clamp applied to both iteration fields. Raise via `Options::with_iteration_cap`. |

`Options` is `#[non_exhaustive]`, so external code must build it from `Options::default()` and assign the fields it wants to change (as above) — a struct literal won't compile outside this crate.

### Output formats

`Format` selects the container for the compressed stream — all readable by standard decoders:

| Variant | Format | Feature |
|---------|--------|---------|
| `Format::Gzip` | gzip ([RFC 1952](https://datatracker.ietf.org/doc/html/rfc1952)) | `gzip` (default) |
| `Format::Zlib` | zlib ([RFC 1950](https://datatracker.ietf.org/doc/html/rfc1950)) | `zlib` (default) |
| `Format::Deflate` | raw DEFLATE ([RFC 1951](https://datatracker.ietf.org/doc/html/rfc1951)) | always available |

### Streaming encoder

```rust
use std::io::Write;

let mut encoder = zenzop::DeflateEncoder::new_buffered(
    zenzop::Options::default(),
    Vec::new(),
);
encoder.write_all(b"Hello, world!").unwrap();
let compressed = encoder.into_inner().unwrap().finish().unwrap().into_inner();
assert!(!compressed.is_empty());
```

### With cancellation / timeout

```rust
use std::io::{self, Write};

fn compress_cancellable(data: &[u8], stop: impl zenzop::Stop) -> io::Result<Vec<u8>> {
    let mut encoder = zenzop::GzipEncoder::with_stop_buffered(
        zenzop::Options::default(),
        Vec::new(),
        stop,
    )?;
    encoder.write_all(data)?;
    let result = encoder.into_inner()?.finish()?;
    if !result.fully_optimized() {
        eprintln!("compression was cut short by the stop token");
    }
    Ok(result.into_inner())
}
```

`Stop`, `StopReason`, and `Unstoppable` are re-exported from [`enough`](https://docs.rs/enough); wire any token (deadline, cancel flag, signal handler) into the `with_stop` / `with_stop_buffered` constructors. `CompressResult::fully_optimized()` tells you whether the encoder finished all iterations or was cut short.

### Command-line tool

The bundled `zenzop` binary compresses each file argument to gzip (`<file>` → `<file>.gz`). It reads two environment variables: `ZENZOP_ITERATIONS` (sets `iteration_count`, default 15) and `ZENZOP_ENHANCED` (any value enables enhanced mode):

```sh
ZENZOP_ENHANCED=1 ZENZOP_ITERATIONS=60 zenzop input.bin   # → input.bin.gz
```

## Cargo features

| Feature | Default | Description |
|---------|---------|-------------|
| `gzip` | yes | Gzip format support |
| `zlib` | yes | Zlib format support |
| `std` | yes | Standard library (logging, `std::io` traits) |
| `parallel` | no | Parallel block compression via rayon |

For `no_std` usage: `default-features = false`. The crate then falls back to a minimal in-crate `Write` trait you can implement for your sink.

## MSRV

The minimum supported Rust version is **1.89**. Bumping this is not considered a breaking change.


## License

Apache-2.0. See [LICENSE](https://github.com/imazen/zenzop/blob/main/LICENSE).

### Upstream contribution

This is a fork of [google/zopfli](https://github.com/google/zopfli) (Apache-2.0). We'd happily release our improvements under the original Apache-2.0 license if upstream wants to take over maintenance — we'd rather contribute back than maintain a parallel codebase. Open an issue or reach out.

### Origin

Forked from [zopfli-rs/zopfli](https://github.com/zopfli-rs/zopfli), Carol Nichols' well-maintained Rust reimplementation of Google's Zopfli. Enhanced-mode optimizations are derived from [ECT](https://github.com/fhanau/Efficient-Compression-Tool) (Efficient Compression Tool) by Felix Hanau. Thanks to both projects — zenzop builds directly on their work.

### AI-generated code notice

Developed with Claude (Anthropic). Not all code has been manually reviewed; review critical paths before production use.

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenjxl-decoder] · [jxl-encoder] · [zenrav1e] · [rav1d-safe] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · **zenzop** · [zenzstd] |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [zenmetrics] · [resamplescope-rs] |
| Pickers & ML | [zenanalyze] · [zenpredict] · [zenpicker] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zentiff
[zenpdf]: https://github.com/imazen/zenpdf
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[zenflate]: https://github.com/imazen/zenflate
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenfilters
[zensally]: https://github.com/imazen/zensally
[zentone]: https://github.com/imazen/zentone
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[zenwasm]: https://github.com/imazen/zenwasm
[zentract]: https://github.com/imazen/zentract
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenmetrics]: https://github.com/imazen/zenmetrics
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[zenanalyze]: https://github.com/imazen/zenanalyze
[zenpredict]: https://github.com/imazen/zenanalyze
[zenpicker]: https://github.com/imazen/zenanalyze
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
