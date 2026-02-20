use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_compress(c: &mut Criterion) {
    let text = std::fs::read("test/data/calgary-books.txt").expect("test data missing");
    let js = std::fs::read("test/data/codetriage.js").expect("test data missing");
    let png = std::fs::read("test/data/computer.png").expect("test data missing");

    let mut group = c.benchmark_group("gzip");
    group.sample_size(10);

    for (name, data) in [("text", &text), ("js", &js), ("png", &png)] {
        group.throughput(Throughput::Bytes(data.len() as u64));

        group.bench_with_input(BenchmarkId::new("zopfli", name), data, |b, data| {
            b.iter(|| {
                let mut out = Vec::with_capacity(data.len());
                zopfli::compress(
                    zopfli::Options::default(),
                    zopfli::Format::Gzip,
                    &data[..],
                    &mut out,
                )
                .unwrap();
                out
            });
        });

        group.bench_with_input(BenchmarkId::new("zenzop", name), data, |b, data| {
            b.iter(|| {
                let mut out = Vec::with_capacity(data.len());
                zenzop::compress(
                    zenzop::Options::default(),
                    zenzop::Format::Gzip,
                    &data[..],
                    &mut out,
                )
                .unwrap();
                out
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_compress);
criterion_main!(benches);
