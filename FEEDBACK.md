# User Feedback Log

## 2026-02-25
- User requested implementation of A1+A2+A3 ECT-derived optimizations behind `enhanced: bool` flag on Options
- User asked "how does this allow ZenPNG to achieve parity with ECT9?" — prompted investigation and iteration
- A3 convergence detection was too aggressive (threshold 2), fixed to match original Zopfli condition
- Milestone RLE at iteration 8 caused regressions on PNG data — removed, keeping only iteration 29 milestone
- Key insight: ECT's advantage comes from Huffman code-length cost model (integer bit lengths) not just entropy
- Ultra post-processing pass with Huffman code-length costs was the biggest single improvement
- Final result: enhanced i60 BEATS ECT-9 by 59 bytes on calgary-books.txt
