#![cfg(feature = "std")]
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use zenzop::{BlockType, DeflateEncoder, Options, Stop, StopReason};

struct StopAfter {
    remaining: AtomicUsize,
    reason: StopReason,
}
impl Stop for StopAfter {
    fn check(&self) -> Result<(), StopReason> {
        if self
            .remaining
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
            .is_err()
        {
            Err(self.reason)
        } else {
            Ok(())
        }
    }
}

#[test]
fn deadlines_finish_valid_streams_and_cancellation_aborts() {
    let input: Vec<u8> = (0..8192).map(|i| ((i * 13 + i / 31) % 251) as u8).collect();
    for block_type in [BlockType::Dynamic, BlockType::Fixed] {
        for after in [0, 1, 3, 12, 40] {
            for reason in [StopReason::TimedOut, StopReason::Cancelled] {
                let mut options = Options::default();
                options.block_type = block_type;
                let stop = StopAfter {
                    remaining: AtomicUsize::new(after),
                    reason,
                };
                let mut encoder = DeflateEncoder::with_stop(options, Vec::new(), stop);
                encoder.write_all(&input).unwrap();
                let result = encoder.finish();
                if reason == StopReason::Cancelled {
                    assert!(result.is_err(), "block={block_type:?}, after={after}");
                } else {
                    let result = result.expect("deadline must still finish the stream");
                    assert!(!result.fully_optimized());
                    assert_eq!(
                        miniz_oxide::inflate::decompress_to_vec(&result.into_inner()).unwrap(),
                        input
                    );
                }
            }
        }
    }
}
