# zenzop

A faster fork of the [Zopfli](https://github.com/google/zopfli) DEFLATE compressor, written in Rust.

Zopfli produces the smallest possible DEFLATE output at the cost of speed. zenzop produces byte-identical output 2-3x faster through algorithmic improvements: precomputed lookup tables, arena-free Huffman tree construction, pre-allocated stores, and eliminated bounds checks in hot paths.

## Features

- **Byte-identical output** to the C Zopfli reference implementation
- **2-3x faster** than the original Rust port
- **`no_std` + `alloc`** compatible — works on embedded and WASM targets
- **Cooperative cancellation** via [`enough::Stop`](https://docs.rs/enough) — cancel long-running compressions cleanly
- **Streaming encoder API** — `DeflateEncoder`, `GzipEncoder`, `ZlibEncoder`
- **Parallel block compression** with optional `rayon` support

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
zenzop = "0.1"
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

### Streaming encoder

```rust
use std::io::{self, Write};

fn main() -> io::Result<()> {
    let mut encoder = zenzop::DeflateEncoder::new_buffered(
        zenzop::Options::default(),
        zenzop::BlockType::Dynamic,
        Vec::new(),
    );
    encoder.write_all(b"Hello, world!")?;
    let compressed = encoder.into_inner()?.finish()?;
    Ok(())
}
```

### With cancellation

```rust
use zenzop::{Options, Format, compress_with_stop, Unstoppable};

// Zero-cost when not needed:
let result = compress_with_stop(Options::default(), Format::Gzip, &data[..], &mut out, Unstoppable);

// With a real stop token (from the `almost-enough` crate):
// let result = compress_with_stop(Options::default(), Format::Gzip, &data[..], &mut out, stop);
```

## Cargo features

| Feature | Default | Description |
|---------|---------|-------------|
| `gzip` | yes | Gzip format support |
| `zlib` | yes | Zlib format support |
| `std` | yes | Standard library (logging, `std::io` traits) |
| `parallel` | no | Parallel block compression via rayon |
| `nightly` | no | Nightly-only optimizations |

For `no_std` usage: `default-features = false`.

## MSRV

The minimum supported Rust version is **1.82**. Bumping this is not considered a breaking change.

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

Forked from [zopfli-rs/zopfli](https://github.com/zopfli-rs/zopfli), which was Carol Nichols' Rust reimplementation of Google's Zopfli.

## AI-Generated Code Notice

Developed with Claude (Anthropic). Not all code manually reviewed. Review critical paths before production use.
