# ABLATION-zenzop.md

**Date:** 2026-06-11  
**Snapshot commit:** f66a6e56 (main@origin)  
**Surface size:** 268 items (default features = all features — identical)  
**Grep template:** `grep -r "<symbol>" /home/lilith/work/zen/zenpng/src --include="*.rs" --exclude-dir=target`

---

## Summary

**0 items flagged. Surface is coherent.**

268 items reviewed. No public-API mistakes found under the conservative bar.

---

## Consumer Evidence

Primary consumer: zenpng (optional `zopfli` feature). All top-level items confirmed consumed:

| Symbol | Confirmed usage |
|---|---|
| `zenzop::Options` | zenpng encoder/compress.rs + examples |
| `zenzop::ZlibEncoder::with_stop` | zenpng encoder/compress.rs |
| `zenzop::compress` | zenpng examples/deflate_bench.rs |
| `zenzop::Format` | zenpng examples |
| `zenzop::Write` (trait) | Implemented for `Vec<u8>` and `&mut [u8]` — consumed implicitly |
| `zenzop::CompressResult` | Returned by encoder `finish()` |
| `zenzop::GzipEncoder` / `zenzop::DeflateEncoder` | Part of surface; zenpng uses ZlibEncoder, the others are parallel API |

---

## Notable design choices (not flagged)

### `Options` struct with public fields

`zenzop::Options` is `#[non_exhaustive]` with all fields public (`block_type`, `enhanced`, `iteration_cap`, `iteration_count`, `iterations_without_improvement`, `maximum_block_splits`). This is the documented configuration API for Zopfli-style optimization parameters — public fields allow direct setting without a builder for each field, and `#[non_exhaustive]` allows future additions. Appropriate for a compression settings struct.

### `DEFAULT_MAX_ITERATIONS` constant

A pub const — confirmed it's referenced in the Options docs/defaults. Zero external callers found in the scan but it's a documentation aid for the iteration tuning surface. Not flagged.

### `zenzop::Write` trait (own variant of `std::io::Write`)

The crate defines its own `Write` trait rather than reusing `std::io::Write`. This is intentional: `enough::Stop`-compatible cancellation needs to thread through the write path, which requires an error type that encodes `StopReason` — not possible with the fixed `std::io::Error` type. The trait also provides the `std::io::Write` bridge impl for `StreamDecompressor<BufReadSource<R>>`. Necessary and well-scoped.

---

## Digest

| Metric | Count |
|---|---|
| Items in surface | 268 |
| Items flagged (Action A) | 0 |
| Items flagged (Action B) | 0 |
| Flag rate | 0% |

**Verdict:** Surface is a tight Zopfli-style encoder API: three format-specific encoders (Deflate/Gzip/Zlib), a `compress` convenience function, an `Options` struct, a `Write` trait, `CompressResult`, and error types. All confirmed consumed. No internal plumbing detected.
