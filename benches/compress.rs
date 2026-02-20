use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::io;

fn compress_data(data: &[u8]) -> Vec<u8> {
    let options = zenzop::Options::default();
    let mut out = Vec::with_capacity(data.len());
    let mut encoder = zenzop::DeflateEncoder::new(options, zenzop::BlockType::Dynamic, &mut out);
    io::copy(&mut &*data, &mut encoder).unwrap();
    encoder.finish().unwrap();
    out
}

fn bench_compress(c: &mut Criterion) {
    let text = std::fs::read("test/data/calgary-books.txt").expect("test data missing");
    let js = std::fs::read("test/data/codetriage.js").expect("test data missing");
    let png = std::fs::read("test/data/computer.png").expect("test data missing");

    let mut group = c.benchmark_group("compress");
    group.sample_size(10);

    for (name, data) in [("text", &text), ("js", &js), ("png", &png)] {
        group.throughput(Throughput::Bytes(data.len() as u64));
        group.bench_with_input(BenchmarkId::new("deflate", name), data, |b, data| {
            b.iter(|| compress_data(data));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_compress);
criterion_main!(benches);
