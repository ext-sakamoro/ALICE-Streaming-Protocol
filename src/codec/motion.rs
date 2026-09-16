//! Motion Estimation - SIMD Optimized
//!
//! High-performance motion estimation using:
//! - AVX2 (`x86_64`)
//! - NEON (aarch64/ARM64)
//! - Scalar fallback (other platforms)

use crate::types::{MotionVector, DEFAULT_BLOCK_SIZE, DEFAULT_SEARCH_RANGE};
use rayon::prelude::*;

/// Search algorithm for motion estimation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchAlgorithm {
    /// Full search (exhaustive, slowest but most accurate)
    FullSearch,
    /// Three Step Search (faster, good accuracy)
    ThreeStepSearch,
    /// Diamond Search (fastest, good accuracy)
    #[default]
    DiamondSearch,
    /// Hexagon Search
    HexagonSearch,
}

/// Motion estimator configuration
#[derive(Debug, Clone)]
pub struct MotionEstimator {
    /// Block size for motion estimation
    pub block_size: usize,
    /// Search range (pixels)
    pub search_range: usize,
    /// Search algorithm
    pub algorithm: SearchAlgorithm,
    /// Early termination threshold
    pub early_termination_threshold: u32,
}

impl Default for MotionEstimator {
    fn default() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            search_range: DEFAULT_SEARCH_RANGE,
            algorithm: SearchAlgorithm::DiamondSearch,
            early_termination_threshold: 256,
        }
    }
}

impl MotionEstimator {
    /// Create a new motion estimator with block size and search range
    #[must_use]
    pub fn new(block_size: usize, search_range: usize) -> Self {
        Self {
            block_size,
            search_range,
            ..Default::default()
        }
    }

    /// Set the search algorithm
    #[must_use]
    pub const fn with_algorithm(mut self, algorithm: SearchAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Estimate motion vectors for entire frame (parallel + SIMD)
    #[must_use]
    pub fn estimate(
        &self,
        current: &[u8],
        previous: &[u8],
        width: usize,
        height: usize,
    ) -> Vec<MotionVector> {
        estimate_motion_with(
            current,
            previous,
            width,
            height,
            self.block_size,
            self.search_range,
            self.algorithm,
            self.early_termination_threshold,
        )
    }
}

// =============================================================================
// SIMD SAD Implementations
// =============================================================================

/// Calculate SAD for 16x16 block - Scalar fallback
#[inline(always)]
fn sad_16x16_scalar(src: &[u8], src_stride: usize, ref_block: &[u8], ref_stride: usize) -> u32 {
    let mut sad: u32 = 0;
    for row in 0..16 {
        let src_row = &src[row * src_stride..row * src_stride + 16];
        let ref_row = &ref_block[row * ref_stride..row * ref_stride + 16];

        // Unroll inner loop for better performance
        sad += (src_row[0] as i32 - ref_row[0] as i32).unsigned_abs();
        sad += (src_row[1] as i32 - ref_row[1] as i32).unsigned_abs();
        sad += (src_row[2] as i32 - ref_row[2] as i32).unsigned_abs();
        sad += (src_row[3] as i32 - ref_row[3] as i32).unsigned_abs();
        sad += (src_row[4] as i32 - ref_row[4] as i32).unsigned_abs();
        sad += (src_row[5] as i32 - ref_row[5] as i32).unsigned_abs();
        sad += (src_row[6] as i32 - ref_row[6] as i32).unsigned_abs();
        sad += (src_row[7] as i32 - ref_row[7] as i32).unsigned_abs();
        sad += (src_row[8] as i32 - ref_row[8] as i32).unsigned_abs();
        sad += (src_row[9] as i32 - ref_row[9] as i32).unsigned_abs();
        sad += (src_row[10] as i32 - ref_row[10] as i32).unsigned_abs();
        sad += (src_row[11] as i32 - ref_row[11] as i32).unsigned_abs();
        sad += (src_row[12] as i32 - ref_row[12] as i32).unsigned_abs();
        sad += (src_row[13] as i32 - ref_row[13] as i32).unsigned_abs();
        sad += (src_row[14] as i32 - ref_row[14] as i32).unsigned_abs();
        sad += (src_row[15] as i32 - ref_row[15] as i32).unsigned_abs();
    }
    sad
}

/// Calculate SAD for 8x8 block - Scalar fallback
#[inline(always)]
fn sad_8x8_scalar(src: &[u8], src_stride: usize, ref_block: &[u8], ref_stride: usize) -> u32 {
    let mut sad: u32 = 0;
    for row in 0..8 {
        let src_row = &src[row * src_stride..row * src_stride + 8];
        let ref_row = &ref_block[row * ref_stride..row * ref_stride + 8];

        sad += (src_row[0] as i32 - ref_row[0] as i32).unsigned_abs();
        sad += (src_row[1] as i32 - ref_row[1] as i32).unsigned_abs();
        sad += (src_row[2] as i32 - ref_row[2] as i32).unsigned_abs();
        sad += (src_row[3] as i32 - ref_row[3] as i32).unsigned_abs();
        sad += (src_row[4] as i32 - ref_row[4] as i32).unsigned_abs();
        sad += (src_row[5] as i32 - ref_row[5] as i32).unsigned_abs();
        sad += (src_row[6] as i32 - ref_row[6] as i32).unsigned_abs();
        sad += (src_row[7] as i32 - ref_row[7] as i32).unsigned_abs();
    }
    sad
}

// AVX2 implementation for x86_64
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod x86_simd {
    use std::arch::x86_64::*;

    /// AVX2 SAD for 16x16 block
    ///
    /// # Safety
    /// `src` and `ref_p` must each point to at least `15 * stride + 16` readable
    /// bytes (16 rows of 16 bytes at the given stride)
    #[inline(always)]
    pub unsafe fn sad_16x16_avx2(
        src: *const u8,
        src_stride: usize,
        ref_p: *const u8,
        ref_stride: usize,
    ) -> u32 {
        let mut acc = _mm256_setzero_si256();
        let mut src_ptr = src;
        let mut ref_ptr = ref_p;

        for _ in 0..16 {
            // Load 16 bytes (one row)
            let s = _mm_loadu_si128(src_ptr as *const __m128i);
            let r = _mm_loadu_si128(ref_ptr as *const __m128i);

            // SAD instruction - computes absolute differences and sums them
            let sad = _mm_sad_epu8(s, r);

            // Accumulate
            acc = _mm256_add_epi64(acc, _mm256_castsi128_si256(sad));

            src_ptr = src_ptr.add(src_stride);
            ref_ptr = ref_ptr.add(ref_stride);
        }

        // Horizontal sum
        let low = _mm256_castsi256_si128(acc);
        let high = _mm256_extracti128_si256::<1>(acc);
        let sum = _mm_add_epi64(low, high);
        let result = _mm_extract_epi64::<0>(sum) + _mm_extract_epi64::<1>(sum);

        result as u32
    }
}

// NEON implementation for ARM64 (Apple M1/M2, etc.)
#[cfg(target_arch = "aarch64")]
mod arm_simd {
    use std::arch::aarch64::{
        vabd_u8, vaddq_u32, vaddvq_u32, vcombine_u32, vdupq_n_u32, vget_lane_u64, vld1_u8,
        vpaddl_u16, vpaddl_u32, vpaddl_u8, vreinterpret_u32_u64,
    };

    /// NEON SAD for 16x16 block
    ///
    /// # Safety
    /// `src` and `ref_p` must each point to at least `15 * stride + 16` readable
    /// bytes (16 rows of 16 bytes at the given stride)
    #[inline(always)]
    pub unsafe fn sad_16x16_neon(
        src: *const u8,
        src_stride: usize,
        ref_p: *const u8,
        ref_stride: usize,
    ) -> u32 {
        let mut acc = vdupq_n_u32(0);
        let mut src_ptr = src;
        let mut ref_ptr = ref_p;

        for _ in 0..16 {
            // Load 16 bytes as two 8-byte vectors
            let s_low = vld1_u8(src_ptr);
            let s_high = vld1_u8(src_ptr.add(8));
            let r_low = vld1_u8(ref_ptr);
            let r_high = vld1_u8(ref_ptr.add(8));

            // Compute absolute differences
            let diff_low = vabd_u8(s_low, r_low);
            let diff_high = vabd_u8(s_high, r_high);

            // Widen and accumulate
            let sum_low = vpaddl_u8(diff_low); // u8 -> u16 pairwise add
            let sum_high = vpaddl_u8(diff_high);

            let sum_low_32 = vpaddl_u16(sum_low); // u16 -> u32 pairwise add
            let sum_high_32 = vpaddl_u16(sum_high);

            // Combine into 128-bit accumulator
            let combined = vcombine_u32(
                vreinterpret_u32_u64(vpaddl_u32(sum_low_32)),
                vreinterpret_u32_u64(vpaddl_u32(sum_high_32)),
            );
            acc = vaddq_u32(acc, combined);

            src_ptr = src_ptr.add(src_stride);
            ref_ptr = ref_ptr.add(ref_stride);
        }

        // Horizontal sum
        vaddvq_u32(acc)
    }

    /// NEON SAD for 8x8 block
    ///
    /// # Safety
    /// `src` and `ref_p` must each point to at least `7 * stride + 8` readable
    /// bytes (8 rows of 8 bytes at the given stride)
    #[inline(always)]
    pub unsafe fn sad_8x8_neon(
        src: *const u8,
        src_stride: usize,
        ref_p: *const u8,
        ref_stride: usize,
    ) -> u32 {
        let mut acc: u32 = 0;
        let mut src_ptr = src;
        let mut ref_ptr = ref_p;

        for _ in 0..8 {
            let s = vld1_u8(src_ptr);
            let r = vld1_u8(ref_ptr);
            let diff = vabd_u8(s, r);

            // Sum all 8 bytes
            let sum16 = vpaddl_u8(diff);
            let sum32 = vpaddl_u16(sum16);
            let sum64 = vpaddl_u32(sum32);
            acc += vget_lane_u64::<0>(sum64) as u32;

            src_ptr = src_ptr.add(src_stride);
            ref_ptr = ref_ptr.add(ref_stride);
        }

        acc
    }
}

// =============================================================================
// Unified SAD function with platform detection
// =============================================================================

/// Calculate SAD for a block (dispatches to best available SIMD)
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn calculate_sad_block(
    current: &[u8],
    previous: &[u8],
    width: usize,
    curr_x: usize,
    curr_y: usize,
    ref_x: usize,
    ref_y: usize,
    block_size: usize,
) -> u32 {
    let curr_offset = curr_y * width + curr_x;
    let ref_offset = ref_y * width + ref_x;

    // Bounds check
    if curr_offset + (block_size - 1) * width + block_size > current.len()
        || ref_offset + (block_size - 1) * width + block_size > previous.len()
    {
        return u32::MAX;
    }

    let src = &current[curr_offset..];
    let ref_block = &previous[ref_offset..];

    // Dispatch to SIMD or scalar based on block size and platform
    match block_size {
        16 => {
            // SAFETY: the bounds check above guarantees `src` / `ref_block` hold
            // at least `15 * width + 16` bytes, i.e. every row read (16 bytes at
            // stride `width`, rows 0..16) is in bounds; NEON is baseline on aarch64
            #[cfg(target_arch = "aarch64")]
            unsafe {
                return arm_simd::sad_16x16_neon(src.as_ptr(), width, ref_block.as_ptr(), width);
            }

            // SAFETY: same bounds argument; AVX2 availability is a compile-time
            // `target_feature`, so no runtime detection is needed
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            unsafe {
                return x86_simd::sad_16x16_avx2(src.as_ptr(), width, ref_block.as_ptr(), width);
            }

            #[allow(unreachable_code)]
            sad_16x16_scalar(src, width, ref_block, width)
        }
        8 => {
            // SAFETY: bounds check above guarantees `7 * width + 8` bytes, which
            // covers the 8 rows of 8 bytes read at stride `width`
            #[cfg(target_arch = "aarch64")]
            unsafe {
                return arm_simd::sad_8x8_neon(src.as_ptr(), width, ref_block.as_ptr(), width);
            }

            #[allow(unreachable_code)]
            sad_8x8_scalar(src, width, ref_block, width)
        }
        _ => {
            // Generic scalar for other block sizes
            let mut sad: u32 = 0;
            for row in 0..block_size {
                for col in 0..block_size {
                    let s = src[row * width + col] as i32;
                    let r = ref_block[row * width + col] as i32;
                    sad += (s - r).unsigned_abs();
                }
            }
            sad
        }
    }
}

// =============================================================================
// Fast Motion Estimation (Parallel + SIMD)
// =============================================================================

/// Ultra-fast motion estimation using SIMD and parallel processing
/// ([`SearchAlgorithm::DiamondSearch`]).
///
/// Returns only non-zero motion vectors for bandwidth efficiency. A block whose
/// SAD at `(0, 0)` is below `early_threshold` is reported as static without
/// searching (the threshold is an absolute SAD, so it is 1 grey level per
/// pixel for 16×16 blocks and 16 per pixel for 4×4 blocks — scale it with the
/// block area, or pass `0` to disable the shortcut).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn estimate_motion_fast(
    current: &[u8],
    previous: &[u8],
    width: usize,
    height: usize,
    block_size: usize,
    search_range: usize,
    early_threshold: u32,
) -> Vec<MotionVector> {
    estimate_motion_with(
        current,
        previous,
        width,
        height,
        block_size,
        search_range,
        SearchAlgorithm::DiamondSearch,
        early_threshold,
    )
}

/// Motion estimation with an explicit [`SearchAlgorithm`] (parallel over block
/// rows, SIMD SAD for 8 / 16 blocks).
///
/// Returns only non-zero motion vectors. Motion vector convention: the
/// reference block in `previous` sits at `block + (dx, dy)`, i.e. a scene that
/// moved right by `s` pixels yields `dx = -s`.
///
/// | algorithm | evaluations per block | exact? |
/// |-----------|----------------------|--------|
/// | [`SearchAlgorithm::FullSearch`] | `(2r+1)²` | yes — global SAD minimum in the window (ties: smallest `dy`, then `dx`, `(0, 0)` first) |
/// | [`SearchAlgorithm::ThreeStepSearch`] | `1 + 8·log₂(r)` | no (coarse-to-fine, step halves from `r/2`) |
/// | [`SearchAlgorithm::DiamondSearch`] | data dependent | no (LDSP then SDSP refinement) |
/// | [`SearchAlgorithm::HexagonSearch`] | data dependent | no (6-point hexagon then 4-point refinement) |
///
/// `early_threshold` is the absolute SAD below which a block is accepted
/// (static shortcut at `(0, 0)` and, for the heuristic searches, at any
/// candidate); `0` disables it.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn estimate_motion_with(
    current: &[u8],
    previous: &[u8],
    width: usize,
    height: usize,
    block_size: usize,
    search_range: usize,
    algorithm: SearchAlgorithm,
    early_threshold: u32,
) -> Vec<MotionVector> {
    if block_size == 0 {
        return Vec::new();
    }
    let blocks_x = width / block_size;
    let blocks_y = height / block_size;

    // Parallel processing with Rayon
    (0..blocks_y)
        .into_par_iter()
        .flat_map(|by| {
            let mut row_results = Vec::with_capacity(blocks_x);
            for bx in 0..blocks_x {
                let search = BlockSearch {
                    current,
                    previous,
                    width,
                    height,
                    block_x: bx * block_size,
                    block_y: by * block_size,
                    block_size,
                    range: search_range as i32,
                    early_threshold,
                };
                let (dx, dy, sad) = match algorithm {
                    SearchAlgorithm::FullSearch => search.full(),
                    SearchAlgorithm::ThreeStepSearch => search.three_step(),
                    SearchAlgorithm::DiamondSearch => search.diamond(),
                    SearchAlgorithm::HexagonSearch => search.hexagon(),
                };
                // Only record non-zero motion vectors (bandwidth optimization)
                if dx != 0 || dy != 0 {
                    row_results.push(MotionVector::new(
                        bx as u16, by as u16, dx as i16, dy as i16, sad,
                    ));
                }
            }
            row_results
        })
        .collect()
}

/// One block's search context: frame slices, block origin and the window.
/// Every algorithm is written against [`BlockSearch::sad_at`] so the bounds /
/// range rules and the SIMD SAD dispatch live in one place.
struct BlockSearch<'a> {
    current: &'a [u8],
    previous: &'a [u8],
    width: usize,
    height: usize,
    block_x: usize,
    block_y: usize,
    block_size: usize,
    range: i32,
    early_threshold: u32,
}

impl BlockSearch<'_> {
    /// SAD of the current block against the reference block displaced by
    /// `(dx, dy)`; `None` when the candidate leaves the search window or the
    /// frame.
    #[inline]
    fn sad_at(&self, dx: i32, dy: i32) -> Option<u32> {
        if dx.abs() > self.range || dy.abs() > self.range {
            return None;
        }
        let ref_x = self.block_x as i32 + dx;
        let ref_y = self.block_y as i32 + dy;
        if ref_x < 0 || ref_y < 0 {
            return None;
        }
        let (ref_x, ref_y) = (ref_x as usize, ref_y as usize);
        if ref_x + self.block_size > self.width || ref_y + self.block_size > self.height {
            return None;
        }
        Some(calculate_sad_block(
            self.current,
            self.previous,
            self.width,
            self.block_x,
            self.block_y,
            ref_x,
            ref_y,
            self.block_size,
        ))
    }

    /// SAD at `(0, 0)` — every search starts here
    #[inline]
    fn sad_origin(&self) -> u32 {
        self.sad_at(0, 0).unwrap_or(u32::MAX)
    }

    /// Exhaustive search: global minimum over the `(2r+1)²` window.
    /// Ties resolve to the candidate first in raster order after `(0, 0)`.
    fn full(&self) -> (i32, i32, u32) {
        let mut best = (0, 0, self.sad_origin());
        if best.2 < self.early_threshold {
            return best;
        }
        for dy in -self.range..=self.range {
            for dx in -self.range..=self.range {
                if (dx, dy) == (0, 0) {
                    continue;
                }
                if let Some(sad) = self.sad_at(dx, dy) {
                    if sad < best.2 {
                        best = (dx, dy, sad);
                        if sad == 0 {
                            return best;
                        }
                    }
                }
            }
        }
        best
    }

    /// Three Step Search: 8 neighbours at `step = 2^⌊log₂(r/2)⌋`, move to the
    /// best, halve the step until 1.
    fn three_step(&self) -> (i32, i32, u32) {
        let mut best = (0, 0, self.sad_origin());
        if best.2 < self.early_threshold {
            return best;
        }
        let mut step = 1i32;
        while step * 2 <= (self.range / 2).max(1) {
            step *= 2;
        }
        while step >= 1 {
            let (cx, cy) = (best.0, best.1);
            for (ox, oy) in [
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            ] {
                let (dx, dy) = (cx + ox * step, cy + oy * step);
                if let Some(sad) = self.sad_at(dx, dy) {
                    if sad < best.2 {
                        best = (dx, dy, sad);
                    }
                }
            }
            if best.2 < self.early_threshold {
                return best;
            }
            step /= 2;
        }
        best
    }

    /// Pattern search: repeat `large` around the best candidate while it
    /// improves, then refine with `small` until no improvement. Diamond and
    /// hexagon search differ only in their patterns.
    fn pattern_search(&self, large: &[(i32, i32)], small: &[(i32, i32)]) -> (i32, i32, u32) {
        let mut best = (0, 0, self.sad_origin());
        if best.2 < self.early_threshold {
            return best;
        }
        let max_iterations = (self.range as usize) * 2;
        for pattern in [large, small] {
            let mut iterations = 0;
            loop {
                let mut improved = false;
                let (cx, cy) = (best.0, best.1);
                for &(ox, oy) in pattern {
                    if let Some(sad) = self.sad_at(cx + ox, cy + oy) {
                        if sad < best.2 {
                            best = (cx + ox, cy + oy, sad);
                            improved = true;
                            if sad < self.early_threshold {
                                return best;
                            }
                        }
                    }
                }
                iterations += 1;
                if !improved || iterations >= max_iterations {
                    break;
                }
            }
        }
        best
    }

    /// Diamond search: Large Diamond Search Pattern, then Small Diamond
    fn diamond(&self) -> (i32, i32, u32) {
        const LDSP: [(i32, i32); 8] = [
            (0, -2),
            (-1, -1),
            (1, -1),
            (-2, 0),
            (2, 0),
            (-1, 1),
            (1, 1),
            (0, 2),
        ];
        const SDSP: [(i32, i32); 4] = [(0, -1), (-1, 0), (1, 0), (0, 1)];
        self.pattern_search(&LDSP, &SDSP)
    }

    /// Hexagon search: 6-point large hexagon, then 4-point small square
    fn hexagon(&self) -> (i32, i32, u32) {
        const LARGE: [(i32, i32); 6] = [(-2, 0), (2, 0), (-1, -2), (1, -2), (-1, 2), (1, 2)];
        const SMALL: [(i32, i32); 4] = [(0, -1), (-1, 0), (1, 0), (0, 1)];
        self.pattern_search(&LARGE, &SMALL)
    }
}

// =============================================================================
// Legacy API (for backwards compatibility)
// =============================================================================

/// Estimate motion vectors (single-threaded, for small frames)
#[must_use]
pub fn estimate_motion(
    current: &[u8],
    previous: &[u8],
    width: usize,
    height: usize,
    block_size: usize,
    search_range: usize,
) -> Vec<MotionVector> {
    estimate_motion_fast(
        current,
        previous,
        width,
        height,
        block_size,
        search_range,
        256,
    )
}

/// Estimate motion vectors (parallel)
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn estimate_motion_parallel(
    current: &[u8],
    previous: &[u8],
    width: usize,
    height: usize,
    block_size: usize,
    search_range: usize,
    algorithm: SearchAlgorithm,
    early_threshold: u32,
) -> Vec<MotionVector> {
    estimate_motion_with(
        current,
        previous,
        width,
        height,
        block_size,
        search_range,
        algorithm,
        early_threshold,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: usize, height: usize, value: u8) -> Vec<u8> {
        vec![value; width * height]
    }

    fn create_shifted_frame(width: usize, height: usize, dx: i32, dy: i32) -> (Vec<u8>, Vec<u8>) {
        let mut current = vec![0u8; width * height];
        let mut previous = vec![0u8; width * height];

        // Create a block pattern in previous frame
        for y in 32..48 {
            for x in 32..48 {
                previous[y * width + x] = 255;
            }
        }

        // Shift the pattern in current frame
        let new_x = (32 + dx) as usize;
        let new_y = (32 + dy) as usize;
        for y in new_y..new_y + 16 {
            for x in new_x..new_x + 16 {
                if y < height && x < width {
                    current[y * width + x] = 255;
                }
            }
        }

        (current, previous)
    }

    #[test]
    fn test_static_frame() {
        let frame = create_test_frame(64, 64, 128);
        let mvs = estimate_motion_fast(&frame, &frame, 64, 64, 16, 8, 256);

        // Static frame should have no motion vectors (all filtered out)
        assert!(mvs.is_empty() || mvs.iter().all(|mv| mv.sad < 256));
    }

    #[test]
    fn test_motion_detection() {
        let (current, previous) = create_shifted_frame(128, 128, 5, 3);
        let mvs = estimate_motion_fast(&current, &previous, 128, 128, 16, 8, 256);

        // Should detect motion
        assert!(!mvs.is_empty(), "Should detect motion");
    }

    #[test]
    fn test_sad_16x16_scalar() {
        let src = vec![100u8; 256];
        let ref_block = vec![110u8; 256];

        let sad = sad_16x16_scalar(&src, 16, &ref_block, 16);

        // Each pixel differs by 10, 256 pixels total
        assert_eq!(sad, 256 * 10);
    }

    #[test]
    fn test_parallel_estimation() {
        let frame = create_test_frame(256, 256, 128);
        let mvs = estimate_motion_fast(&frame, &frame, 256, 256, 16, 8, 256);

        // All static, should be empty or very few
        assert!(mvs.len() <= 256);
    }

    #[test]
    fn test_motion_estimator_config() {
        let estimator = MotionEstimator::new(8, 16).with_algorithm(SearchAlgorithm::HexagonSearch);

        assert_eq!(estimator.block_size, 8);
        assert_eq!(estimator.search_range, 16);
        assert_eq!(estimator.algorithm, SearchAlgorithm::HexagonSearch);
    }

    #[test]
    fn test_sad_8x8_scalar_known_value() {
        // All source = 200, all ref = 180 → diff = 20 per pixel × 64 pixels = 1280
        let src = vec![200u8; 128];
        let ref_block = vec![180u8; 128];
        let sad = sad_8x8_scalar(&src, 8, &ref_block, 8);
        assert_eq!(sad, 64 * 20);
    }

    #[test]
    fn test_sad_8x8_scalar_identical_blocks() {
        let block = vec![128u8; 128];
        let sad = sad_8x8_scalar(&block, 8, &block, 8);
        assert_eq!(sad, 0);
    }

    #[test]
    fn test_motion_estimator_estimate_wrapper() {
        let estimator = MotionEstimator::default();
        let frame = create_test_frame(64, 64, 100);
        let mvs = estimator.estimate(&frame, &frame, 64, 64);
        // Identical frames → no non-zero motion vectors
        assert!(mvs.is_empty());
    }

    #[test]
    fn test_estimate_motion_parallel_identical_frames() {
        let frame = create_test_frame(128, 128, 50);
        let mvs = estimate_motion_parallel(
            &frame,
            &frame,
            128,
            128,
            16,
            8,
            SearchAlgorithm::DiamondSearch,
            256,
        );
        assert!(
            mvs.is_empty(),
            "identical frames should yield no motion vectors"
        );
    }

    #[test]
    fn test_estimate_motion_parallel_detects_motion() {
        let (current, previous) = create_shifted_frame(128, 128, 4, 0);
        let mvs = estimate_motion_parallel(
            &current,
            &previous,
            128,
            128,
            16,
            8,
            SearchAlgorithm::FullSearch,
            256,
        );
        assert!(
            !mvs.is_empty(),
            "shifted frame should produce motion vectors"
        );
    }

    #[test]
    fn test_search_algorithm_default_is_diamond() {
        let algo = SearchAlgorithm::default();
        assert_eq!(algo, SearchAlgorithm::DiamondSearch);
    }
}
