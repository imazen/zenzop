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

# Format code + regenerate the public-API surface snapshot (docs/public-api/)
fmt:
    cargo fmt
    cargo test -p zenzop --test public_api_doc

# Regenerate the public-API surface snapshot only
api-doc:
    cargo test -p zenzop --test public_api_doc

# Verify the committed snapshot is current (what CI runs)
api-doc-check:
    ZEN_API_DOC=check cargo test -p zenzop --test public_api_doc

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
