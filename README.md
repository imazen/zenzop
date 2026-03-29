# zenzop [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenzop/ci.yml?style=flat-square)](https://github.com/imazen/zenzop/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zenzop?style=flat-square)](https://crates.io/crates/zenzop) [![lib.rs](https://img.shields.io/crates/v/zenzop?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenzop) [![docs.rs](https://img.shields.io/docsrs/zenzop?style=flat-square)](https://docs.rs/zenzop) [![license](https://img.shields.io/crates/l/zenzop?style=flat-square)](https://github.com/imazen/zenzop#license)

A faster fork of the [Zopfli](https://github.com/google/zopfli) DEFLATE compressor, written in Rust.

Zopfli produces near-optimal DEFLATE output at the cost of speed. zenzop produces byte-identical output 1.2–2x faster through algorithmic improvements: precomputed cost tables, SIMD-accelerated match comparison, arena-based Huffman tree construction, pre-allocated stores, and a skip-hash optimization that eliminates redundant hash chain walks on cached iterations.

With `enhanced` mode enabled, zenzop applies ECT-derived optimizations — expanded precode search, multi-strategy Huffman tree selection, and enhanced parser diversification — to produce smaller output than standard Zopfli at the cost of byte-for-byte parity with the C reference.

## Features

- **Byte-identical output** to C Zopfli (default mode)
- **Enhanced mode** for smaller-than-Zopfli output (beats ECT-9 at equivalent iterations)
- **1.2–2x faster** than C Zopfli (input-dependent; larger gains on smaller blocks)
- **`no_std` + `alloc`** compatible — works on embedded and WASM targets
- **Cooperative cancellation** via [`enough::Stop`](https://docs.rs/enough) — cancel long-running compressions cleanly
- **Streaming encoder API** — `DeflateEncoder`, `GzipEncoder`, `ZlibEncoder`
- **Parallel block compression** with optional `rayon` support

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
zenzop = "0.4"
```

### Compress data

```rust
use std::io;

fn main() -> io::Result<()> {
    let data = b"Hello, world!";
    let mut output = Vec::new();

    zenzop::compress(
        zenzop::Options::default(),
        zenzop::Format::Gzip,
        &data[..],
        &mut output,
    )?;

    Ok(())
}
```

### Enhanced mode

```rust
use std::io;

fn main() -> io::Result<()> {
    let data = b"Hello, world!";
    let mut output = Vec::new();

    let mut options = zenzop::Options::default();
    options.enhanced = true;

    zenzop::compress(options, zenzop::Format::Gzip, &data[..], &mut output)?;

    Ok(())
}
```

Enhanced mode produces smaller DEFLATE output than standard Zopfli with ~5% runtime overhead. At 60 iterations it beats ECT-9 on representative test data.

### Streaming encoder

```rust
use std::io::{self, Write};

fn main() -> io::Result<()> {
    let mut encoder = zenzop::DeflateEncoder::new_buffered(
        zenzop::Options::default(),
        Vec::new(),
    );
    encoder.write_all(b"Hello, world!")?;
    let compressed = encoder.into_inner()?.finish()?.into_inner();
    Ok(())
}
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
        eprintln!("compression was cut short by timeout");
    }
    Ok(result.into_inner())
}
```

## Cargo features

| Feature | Default | Description |
|---------|---------|-------------|
| `gzip` | yes | Gzip format support |
| `zlib` | yes | Zlib format support |
| `std` | yes | Standard library (logging, `std::io` traits) |
| `parallel` | no | Parallel block compression via rayon |

For `no_std` usage: `default-features = false`.

## MSRV

The minimum supported Rust version is **1.89**. Bumping this is not considered a breaking change.

## Building from source

```
cargo build --release
```

The `zenzop` binary will be at `target/release/zenzop`.

## Testing

```
cargo test                    # Unit tests + property-based tests
./test/run.sh                 # Golden master: byte-identical to C Zopfli
```

## Image tech I maintain

| | |
|:--|:--|
| State of the art codecs<sup>[1]</sup> | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] ([rav1d-safe] · [zenrav1e] · [zenavif-parse] · [zenavif-serialize]) · [zenjxl] ([jxl-encoder] · [zenjxl-decoder]) · [zentiff] · [zenbitmaps] · [heic] · [zenraw] · [zenpdf] · [ultrahdr] · [mozjpeg-rs] · [webpx] |
| Compression | [zenflate] · **zenzop** |
| Processing | [zenresize] · [zenfilters] · [zenquant] · [zenblend] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [resamplescope-rs] · [codec-eval] · [codec-corpus] |
| Pixel types & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] |
| ImageResizer | [ImageResizer] (C#) — 24M+ NuGet downloads across all packages |
| [Imageflow][] | Image optimization engine (Rust) — [.NET][imageflow-dotnet] · [node][imageflow-node] · [go][imageflow-go] — 9M+ NuGet downloads across all packages |
| [Imageflow Server][] | [The fast, safe image server](https://www.imazen.io/) (Rust+C#) — 552K+ NuGet downloads, deployed by Fortune 500s and major brands |

<sup>[1]</sup> <sub>as of 2026</sub>

### General Rust awesomeness

[archmage] · [magetypes] · [enough] · [whereat] · [zenbench] · [cargo-copter]

[And other projects](https://www.imazen.io/open-source) · [GitHub @imazen](https://github.com/imazen) · [GitHub @lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith) · [NuGet](https://www.nuget.org/profiles/imazen) (over 30 million downloads / 87 packages)

[zenjpeg]: https://crates.io/crates/zenjpeg
[zenpng]: https://crates.io/crates/zenpng
[zenwebp]: https://crates.io/crates/zenwebp
[zengif]: https://crates.io/crates/zengif
[zenavif]: https://crates.io/crates/zenavif
[rav1d-safe]: https://crates.io/crates/rav1d-safe
[zenrav1e]: https://crates.io/crates/zenrav1e
[zenavif-parse]: https://crates.io/crates/zenavif-parse
[zenavif-serialize]: https://crates.io/crates/zenavif-serialize
[zenjxl]: https://crates.io/crates/zenjxl
[jxl-encoder]: https://crates.io/crates/jxl-encoder
[zenjxl-decoder]: https://crates.io/crates/zenjxl-decoder
[zentiff]: https://crates.io/crates/zentiff
[zenbitmaps]: https://crates.io/crates/zenbitmaps
[heic]: https://crates.io/crates/heic
[zenraw]: https://crates.io/crates/zenraw
[zenpdf]: https://crates.io/crates/zenpdf
[ultrahdr]: https://crates.io/crates/ultrahdr
[mozjpeg-rs]: https://crates.io/crates/mozjpeg-rs
[webpx]: https://crates.io/crates/webpx
[zenflate]: https://crates.io/crates/zenflate
[zenresize]: https://crates.io/crates/zenresize
[zenfilters]: https://crates.io/crates/zenfilters
[zenquant]: https://crates.io/crates/zenquant
[zenblend]: https://crates.io/crates/zenblend
[zensim]: https://crates.io/crates/zensim
[fast-ssim2]: https://crates.io/crates/fast-ssim2
[butteraugli]: https://crates.io/crates/butteraugli
[resamplescope-rs]: https://crates.io/crates/resamplescope-rs
[codec-eval]: https://crates.io/crates/codec-eval
[codec-corpus]: https://crates.io/crates/codec-corpus
[zenpixels]: https://crates.io/crates/zenpixels
[zenpixels-convert]: https://crates.io/crates/zenpixels-convert
[linear-srgb]: https://crates.io/crates/linear-srgb
[garb]: https://crates.io/crates/garb
[zenpipe]: https://crates.io/crates/zenpipe
[zencodec]: https://crates.io/crates/zencodec
[zencodecs]: https://crates.io/crates/zencodecs
[zenlayout]: https://crates.io/crates/zenlayout
[zennode]: https://crates.io/crates/zennode
[ImageResizer]: https://imageresizing.net
[Imageflow]: https://github.com/imazen/imageflow
[imageflow-dotnet]: https://www.nuget.org/packages/Imageflow.AllPlatforms
[imageflow-node]: https://www.npmjs.com/package/@imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[archmage]: https://crates.io/crates/archmage
[magetypes]: https://crates.io/crates/magetypes
[enough]: https://crates.io/crates/enough
[whereat]: https://crates.io/crates/whereat
[zenbench]: https://crates.io/crates/zenbench
[cargo-copter]: https://crates.io/crates/cargo-copter

## License

Apache-2.0

### Upstream Contribution

This is a fork of [google/zopfli](https://github.com/google/zopfli) (Apache-2.0).
We are willing to release our improvements under the original Apache-2.0
license if upstream takes over maintenance of those improvements. We'd rather
contribute back than maintain a parallel codebase. Open an issue or reach out.

## Origin

Forked from [zopfli-rs/zopfli](https://github.com/zopfli-rs/zopfli), which was Carol Nichols' Rust reimplementation of Google's Zopfli. Enhanced mode optimizations derived from [ECT](https://github.com/fhanau/Efficient-Compression-Tool) (Efficient Compression Tool).

## AI-Generated Code Notice

Developed with Claude (Anthropic). Not all code manually reviewed. Review critical paths before production use.
