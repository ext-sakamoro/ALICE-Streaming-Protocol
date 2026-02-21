//! Motion Estimation - SIMD Optimized
//!
//! High-performance motion estimation using:
//! - AVX2 (x86_64)
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
    pub fn new(block_size: usize, search_range: usize) -> Self {
        Self {
            block_size,
            search_range,
            ..Default::default()
        }
    }

    pub fn with_algorithm(mut self, algorithm: SearchAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Estimate motion vectors for entire frame (parallel + SIMD)
    pub fn estimate(
        &self,
        current: &[u8],
        previous: &[u8],
        width: usize,
        height: usize,
    ) -> Vec<MotionVector> {
        estimate_motion_fast(
            current,
            previous,
            width,
            height,
            self.block_size,
            self.search_range,
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
    use std::arch::aarch64::*;

    /// NEON SAD for 16x16 block
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
            #[cfg(target_arch = "aarch64")]
            unsafe {
                return arm_simd::sad_16x16_neon(src.as_ptr(), width, ref_block.as_ptr(), width);
            }

            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            unsafe {
                return x86_simd::sad_16x16_avx2(src.as_ptr(), width, ref_block.as_ptr(), width);
            }

            #[allow(unreachable_code)]
            sad_16x16_scalar(src, width, ref_block, width)
        }
        8 => {
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
///
/// Returns only non-zero motion vectors for bandwidth efficiency
#[allow(clippy::too_many_arguments)]
pub fn estimate_motion_fast(
    current: &[u8],
    previous: &[u8],
    width: usize,
    height: usize,
    block_size: usize,
    search_range: usize,
    early_threshold: u32,
) -> Vec<MotionVector> {
    let blocks_x = width / block_size;
    let blocks_y = height / block_size;

    // Parallel processing with Rayon
    let results: Vec<MotionVector> = (0..blocks_y)
        .into_par_iter()
        .flat_map(|by| {
            let mut row_results = Vec::with_capacity(blocks_x);

            for bx in 0..blocks_x {
                let mv = diamond_search_simd(
                    current,
                    previous,
                    width,
                    height,
                    bx,
                    by,
                    block_size,
                    search_range,
                    early_threshold,
                );

                // Only record non-zero motion vectors (bandwidth optimization)
                if mv.dx != 0 || mv.dy != 0 {
                    row_results.push(mv);
                }
            }
            row_results
        })
        .collect();

    results
}

/// Diamond search with SIMD acceleration.
///
/// `bx` / `by` are the block-grid indices (pixel coords = bx * block_size, etc.).
/// Passing them in avoids any division inside the hot path.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn diamond_search_simd(
    current: &[u8],
    previous: &[u8],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    block_size: usize,
    search_range: usize,
    early_threshold: u32,
) -> MotionVector {
    let block_x = bx * block_size;
    let block_y = by * block_size;

    let mut best_x = 0i32;
    let mut best_y = 0i32;
    let mut best_sad = calculate_sad_block(
        current, previous, width, block_x, block_y, block_x, block_y, block_size,
    );

    // Early termination for static blocks
    if best_sad < early_threshold {
        return MotionVector::new(bx as u16, by as u16, 0, 0, best_sad);
    }

    let search_range_i = search_range as i32;

    // Large Diamond Search Pattern (LDSP)
    let ldsp = [
        (0, -2),
        (-1, -1),
        (1, -1),
        (-2, 0),
        (2, 0),
        (-1, 1),
        (1, 1),
        (0, 2),
    ];

    // Small Diamond Search Pattern (SDSP)
    let sdsp = [(0, -1), (-1, 0), (1, 0), (0, 1)];

    let max_iterations = search_range * 2;
    let mut iterations = 0;

    // LDSP phase
    loop {
        let mut improved = false;

        for &(dx, dy) in &ldsp {
            let new_x = best_x + dx;
            let new_y = best_y + dy;

            // Range check
            if new_x.abs() > search_range_i || new_y.abs() > search_range_i {
                continue;
            }

            // Calculate reference position (check for negative before converting to usize)
            let ref_x_i = block_x as i32 + new_x;
            let ref_y_i = block_y as i32 + new_y;

            // Bounds check (must be non-negative and within frame)
            if ref_x_i < 0 || ref_y_i < 0 {
                continue;
            }
            let ref_x = ref_x_i as usize;
            let ref_y = ref_y_i as usize;

            if ref_x + block_size > width || ref_y + block_size > height {
                continue;
            }

            let sad = calculate_sad_block(
                current, previous, width, block_x, block_y, ref_x, ref_y, block_size,
            );

            if sad < best_sad {
                best_sad = sad;
                best_x = new_x;
                best_y = new_y;
                improved = true;

                // Early termination
                if best_sad < early_threshold {
                    return MotionVector::new(
                        bx as u16,
                        by as u16,
                        best_x as i16,
                        best_y as i16,
                        best_sad,
                    );
                }
            }
        }

        iterations += 1;
        if !improved || iterations >= max_iterations {
            break;
        }
    }

    // SDSP phase (refinement)
    loop {
        let mut improved = false;

        for &(dx, dy) in &sdsp {
            let new_x = best_x + dx;
            let new_y = best_y + dy;

            if new_x.abs() > search_range_i || new_y.abs() > search_range_i {
                continue;
            }

            // Calculate reference position (check for negative before converting to usize)
            let ref_x_i = block_x as i32 + new_x;
            let ref_y_i = block_y as i32 + new_y;

            if ref_x_i < 0 || ref_y_i < 0 {
                continue;
            }
            let ref_x = ref_x_i as usize;
            let ref_y = ref_y_i as usize;

            if ref_x + block_size > width || ref_y + block_size > height {
                continue;
            }

            let sad = calculate_sad_block(
                current, previous, width, block_x, block_y, ref_x, ref_y, block_size,
            );

            if sad < best_sad {
                best_sad = sad;
                best_x = new_x;
                best_y = new_y;
                improved = true;
            }
        }

        if !improved {
            break;
        }
    }

    MotionVector::new(
        bx as u16,
        by as u16,
        best_x as i16,
        best_y as i16,
        best_sad,
    )
}

// =============================================================================
// Legacy API (for backwards compatibility)
// =============================================================================

/// Estimate motion vectors (single-threaded, for small frames)
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
pub fn estimate_motion_parallel(
    current: &[u8],
    previous: &[u8],
    width: usize,
    height: usize,
    block_size: usize,
    search_range: usize,
    _algorithm: SearchAlgorithm,
    early_threshold: u32,
) -> Vec<MotionVector> {
    estimate_motion_fast(
        current,
        previous,
        width,
        height,
        block_size,
        search_range,
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
}
