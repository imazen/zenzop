# zenzop

[![CI](https://github.com/imazen/zenzop/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/zenzop/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/zenzop.svg)](https://crates.io/crates/zenzop)
[![docs.rs](https://docs.rs/zenzop/badge.svg)](https://docs.rs/zenzop)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![MSRV: 1.89](https://img.shields.io/badge/MSRV-1.89-blue.svg)](https://blog.rust-lang.org/)

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
zenzop = "0.3"
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

## License

Apache-2.0

## Origin

Forked from [zopfli-rs/zopfli](https://github.com/zopfli-rs/zopfli), which was Carol Nichols' Rust reimplementation of Google's Zopfli. Enhanced mode optimizations derived from [ECT](https://github.com/fhanau/Efficient-Compression-Tool) (Efficient Compression Tool).

## AI-Generated Code Notice

Developed with Claude (Anthropic). Not all code manually reviewed. Review critical paths before production use.
