# zenzop benchmarks

zenzop's headline claim is **1.2–2x faster than Zopfli for byte-identical output**.
This directory documents how to reproduce that measurement. We do **not** commit
pre-baked numbers: the ratio depends on the input and the host CPU, so the
harness is the source of truth — run it on your own data and hardware.

The benchmark lives at [`benches/compress.rs`](../benches/compress.rs) and uses
[zenbench](https://crates.io/crates/zenbench) (via its criterion-compat shim),
not criterion directly.

## What it measures

For each of three representative inputs, it times two contenders compressing the
same bytes to the same container, then reports throughput:

| Contender | Crate | Version | Settings |
|-----------|-------|---------|----------|
| `zenzop`  | this crate | (the checked-out commit) | `Options::default()`, `Format::Gzip` |
| `zopfli`  | [`zopfli`](https://crates.io/crates/zopfli) | `0.8.x` (dev-dependency) | `Options::default()`, `Format::Gzip` |

Inputs (in `test/data/`, identical bytes for both contenders):

| Name | File | Kind |
|------|------|------|
| `text` | `calgary-books.txt` | English prose (Calgary corpus) |
| `js`   | `codetriage.js`     | minified-ish JavaScript |
| `png`  | `computer.png`      | a PNG file (already-compressed bytes) |

The `zopfli` crate is a faithful Rust reimplementation of Google's Zopfli and
produces byte-identical DEFLATE, so it is the fair same-algorithm baseline for a
speed comparison. zenzop's default mode is verified byte-identical to C Zopfli by
the golden-master test (`test/run.sh`) in CI.

## Fair-benchmark properties (per the zen* benchmark conventions)

1. **I/O is excluded from the timed region.** Each input file is read once with
   `std::fs::read` *before* the benchmark group; the timed `b.iter(..)` closure
   only compresses from an in-memory `&[u8]` into a `Vec<u8>`. No file open/read
   happens inside the measured loop.
2. **Output is consumed.** The closure returns the compressed `Vec`, so the
   compression can't be optimized away.
3. **Apples-to-apples.** Same images, same dimensions/bytes, same `Format::Gzip`,
   same default options across both contenders.
4. **Threading.** Both run single-threaded: zenzop's `parallel` (rayon) feature
   is off by default, and the `zopfli` crate compresses on one thread. Do not
   compare a parallel build of one against a single-threaded build of the other.
5. **No `-C target-cpu=native`.** Runtime SIMD dispatch is what ships; building
   with `native` bakes in ISA extensions and produces misleading numbers. Build
   with the default profile.

## Reproduce

```sh
git clone https://github.com/imazen/zenzop && cd zenzop
git checkout <commit-sha>           # record the SHA you measured
cargo bench --bench compress        # default profile — no RUSTFLAGS="-C target-cpu=native"
```

Record, alongside any numbers you publish:

- **Environment** — CPU model, RAM, OS, `rustc -V`, and `cargo --version`.
- The **commit SHA** of zenzop and the resolved `zopfli` version (`cargo tree -p zopfli`).
- The **build profile** (this repo's `[profile.bench]` keeps debuginfo on; it does
  not change codegen opts).

## Measuring the compression-ratio tradeoff

The speed bench above fixes `iteration_count` at the default (15). To study the
**size-vs-time tradeoff** that `Options::iteration_count` controls, sweep it and
record both compressed size and wall time per setting — e.g. iterations
`5, 10, 15, 30, 60, 120` with `enhanced` on and off — then plot compressed bytes
(y) against time (x), one line per mode. That curve, not a single number, is the
honest picture of what enhanced mode and extra iterations buy on a given input.

## Charts

Per the zen* conventions:

- **"Which is fastest?"** → horizontal bar chart sorted by throughput (MB/s), one
  bar per contender. zenbench's sorted output or `--format=html` report works.
- **"Size vs time?"** → scatter/line of compressed bytes vs wall time, swept over
  `iteration_count` (see above).

Avoid pie/3D/dual-axis charts — they obscure the comparison.
