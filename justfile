default:
    @just --list

# Run all tests (unit + golden master)
test:
    cargo test
    cargo build --release
    bash test/run.sh

# Run criterion benchmarks
bench:
    cargo bench --bench compress

# Run clippy
lint:
    cargo clippy --all-targets

# Format code
fmt:
    cargo fmt

# Check formatting
fmt-check:
    cargo fmt --check

# Feature permutation checks
feature-check:
    cargo test --features parallel
    cargo test --no-default-features --features "gzip,zlib" --lib
    cargo check --no-default-features

# Run all CI checks locally
ci: fmt-check lint test feature-check

# Profile with callgrind (compresses codetriage.js)
profile:
    cargo build --release
    valgrind --tool=callgrind --callgrind-out-file=/tmp/callgrind.out target/release/zenzop test/data/codetriage.js
    @echo "Output: /tmp/callgrind.out — open with kcachegrind"

# Run golden master tests only
golden:
    cargo build --release
    bash test/run.sh
