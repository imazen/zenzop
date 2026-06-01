# Changelog

## Unreleased

### Internal
- Excluded `.cargo/`, `.gitattributes`, `.rustfmt.toml`, and `benches/` from the published crate tarball (packaging hygiene).

### Added
- `Options::iteration_cap` field, `Options::with_iteration_cap`,
  `Options::effective_iteration_count`, `Options::effective_iterations_without_improvement`,
  and `pub const DEFAULT_MAX_ITERATIONS = 1000`. The encoder now silently clamps both
  `iteration_count` and `iterations_without_improvement` to `iteration_cap` per block.
  Real-world values (5-15) are unchanged; pathological `u64::MAX` configurations now
  terminate in finite time. Callers that genuinely need more iterations can raise the cap
  with `Options::default().with_iteration_cap(n)`.

### Fixed
- Compute-DoS footgun: previously, an `Options` with `iteration_count = u64::MAX` (or
  `iterations_without_improvement = u64::MAX`, the default) and the default unstoppable
  `Stop` would loop until heat death. The internal cap turns that into a finite worst case.

## 0.4.1 (2026-03-25)

### Changed
- Bumped `enough` 0.4 → 0.4.2
- Bumped `zenflate` 0.3 → 0.3.0 (full version specifier)

### Internal
- Added justfile
- Commented out nightly-only rustfmt options

## 0.4.0 (2026-03-05)

Initial public release.
