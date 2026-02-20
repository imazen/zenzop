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

# Profile with callgrind (compresses codetriage.js)
profile:
    cargo build --release
    valgrind --tool=callgrind --callgrind-out-file=/tmp/callgrind.out target/release/zenzop test/data/codetriage.js
    @echo "Output: /tmp/callgrind.out — open with kcachegrind"

# Run golden master tests only
golden:
    cargo build --release
    bash test/run.sh
