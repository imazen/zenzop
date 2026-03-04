use enough::Unstoppable;
use std::time::Instant;

fn codec_corpus_dir() -> std::path::PathBuf {
    let dir = std::path::PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR").unwrap_or_else(|_| "/home/lilith/work/codec-corpus".into()),
    );
    assert!(dir.is_dir(), "Codec corpus not found: {}. Set CODEC_CORPUS_DIR.", dir.display());
    dir
}

fn main() {
    // Use a representative PNG IDAT stream - decompress from a real PNG
    let path = codec_corpus_dir()
        .join("clic2025-1024/0d154749c7771f58e89ad343653ec4e20d6f037da829f47f5598e5d0a4ab61f0.png");
    let png = std::fs::read(path).unwrap();
    let idat = extract_idat(&png);
    let filtered = decompress_zlib(&idat);

    eprintln!("Input: {} bytes filtered data", filtered.len());

    let iters: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);
    let mut options = zenzop::Options::default();
    options.iteration_count = core::num::NonZeroU64::new(iters).unwrap();
    options.enhanced = true;

    let start = Instant::now();
    let mut encoder = zenzop::ZlibEncoder::with_stop(options, Vec::new(), Unstoppable).unwrap();
    std::io::Write::write_all(&mut encoder, &filtered).unwrap();
    let result = encoder.finish().unwrap();
    let len = result.into_inner().len();
    let elapsed = start.elapsed();
    eprintln!(
        "Output: {len} bytes in {:.2}s ({iters}i)",
        elapsed.as_secs_f64()
    );
}

fn extract_idat(png: &[u8]) -> Vec<u8> {
    let mut idat = Vec::new();
    let mut pos = 8;
    while pos + 12 <= png.len() {
        let len = u32::from_be_bytes([png[pos], png[pos + 1], png[pos + 2], png[pos + 3]]) as usize;
        let chunk_type = &png[pos + 4..pos + 8];
        if chunk_type == b"IDAT" {
            idat.extend_from_slice(&png[pos + 8..pos + 8 + len]);
        }
        pos += 12 + len;
    }
    idat
}

fn decompress_zlib(data: &[u8]) -> Vec<u8> {
    let mut d = zenflate::Decompressor::new();
    let mut out = vec![0u8; data.len() * 10];
    loop {
        match d.zlib_decompress(data, &mut out, Unstoppable) {
            Ok(o) => {
                out.truncate(o.output_written);
                return out;
            }
            Err(zenflate::DecompressionError::InsufficientSpace) => {
                out.resize(out.len() * 2, 0);
            }
            Err(e) => panic!("{e}"),
        }
    }
}
