//! Discrete Cosine Transform (DCT)
//!
//! High-performance 2D DCT implementation for video compression.
//! Supports sparse coefficient encoding for ultra-low bandwidth.

use rayon::prelude::*;
use std::f64::consts::PI;

/// DCT Transform configuration
#[derive(Debug, Clone)]
pub struct DctTransform {
    /// Block size (typically 8x8)
    pub block_size: usize,
    /// Quantization matrix
    pub quant_matrix: Vec<f64>,
    /// Sparsity threshold
    pub sparsity_threshold: f64,
}

impl Default for DctTransform {
    fn default() -> Self {
        Self::new(8)
    }
}

impl DctTransform {
    /// Create new DCT transform with specified block size
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            quant_matrix: default_quantization_matrix(block_size),
            sparsity_threshold: 0.001,
        }
    }

    /// Set quantization quality (1-100)
    pub fn with_quality(mut self, quality: u8) -> Self {
        let quality = quality.clamp(1, 100) as f64;
        let scale = if quality < 50.0 {
            5000.0 / quality
        } else {
            200.0 - quality * 2.0
        };

        for q in &mut self.quant_matrix {
            *q = (*q * scale / 100.0).max(1.0);
        }
        self
    }

    /// Forward DCT on a block
    pub fn forward(&self, block: &[f64]) -> Vec<f64> {
        dct2d(block, self.block_size)
    }

    /// Inverse DCT on a block
    pub fn inverse(&self, coefficients: &[f64]) -> Vec<f64> {
        idct2d(coefficients, self.block_size)
    }

    /// Quantize DCT coefficients
    pub fn quantize(&self, coefficients: &[f64]) -> Vec<i32> {
        coefficients
            .iter()
            .zip(self.quant_matrix.iter())
            .map(|(&c, &q)| (c / q).round() as i32)
            .collect()
    }

    /// Dequantize coefficients
    pub fn dequantize(&self, quantized: &[i32]) -> Vec<f64> {
        quantized
            .iter()
            .zip(self.quant_matrix.iter())
            .map(|(&c, &q)| c as f64 * q)
            .collect()
    }

    /// Encode to sparse representation
    pub fn encode_sparse(&self, block: &[f64]) -> Vec<(u32, u32, f32)> {
        let coefficients = self.forward(block);
        let quantized = self.quantize(&coefficients);
        sparse_dct_encode(&quantized, self.block_size, self.sparsity_threshold)
    }

    /// Decode from sparse representation
    pub fn decode_sparse(&self, sparse: &[(u32, u32, f32)], default_value: f64) -> Vec<f64> {
        let quantized = sparse_dct_decode(sparse, self.block_size, default_value as i32);
        let dequantized = self.dequantize(&quantized);
        self.inverse(&dequantized)
    }
}

/// Default JPEG-like quantization matrix
fn default_quantization_matrix(size: usize) -> Vec<f64> {
    if size == 8 {
        // Standard JPEG luminance quantization matrix
        vec![
            16.0, 11.0, 10.0, 16.0, 24.0, 40.0, 51.0, 61.0, 12.0, 12.0, 14.0, 19.0, 26.0, 58.0,
            60.0, 55.0, 14.0, 13.0, 16.0, 24.0, 40.0, 57.0, 69.0, 56.0, 14.0, 17.0, 22.0, 29.0,
            51.0, 87.0, 80.0, 62.0, 18.0, 22.0, 37.0, 56.0, 68.0, 109.0, 103.0, 77.0, 24.0, 35.0,
            55.0, 64.0, 81.0, 104.0, 113.0, 92.0, 49.0, 64.0, 78.0, 87.0, 103.0, 121.0, 120.0,
            101.0, 72.0, 92.0, 95.0, 98.0, 112.0, 100.0, 103.0, 99.0,
        ]
    } else {
        // Generate a generic quantization matrix
        (0..size * size)
            .map(|i| {
                let x = i % size;
                let y = i / size;
                ((x + y + 2) * 4) as f64
            })
            .collect()
    }
}

/// 2D Discrete Cosine Transform
pub fn dct2d(input: &[f64], size: usize) -> Vec<f64> {
    let mut output = vec![0.0; size * size];
    let scale = 2.0 / size as f64;

    for v in 0..size {
        for u in 0..size {
            let cu = if u == 0 { 1.0 / 2.0_f64.sqrt() } else { 1.0 };
            let cv = if v == 0 { 1.0 / 2.0_f64.sqrt() } else { 1.0 };

            let mut sum = 0.0;
            for y in 0..size {
                for x in 0..size {
                    let cos_u = ((2 * x + 1) as f64 * u as f64 * PI / (2.0 * size as f64)).cos();
                    let cos_v = ((2 * y + 1) as f64 * v as f64 * PI / (2.0 * size as f64)).cos();
                    sum += input[y * size + x] * cos_u * cos_v;
                }
            }

            output[v * size + u] = scale * cu * cv * sum;
        }
    }

    output
}

/// 2D Inverse Discrete Cosine Transform
pub fn idct2d(input: &[f64], size: usize) -> Vec<f64> {
    let mut output = vec![0.0; size * size];
    let scale = 2.0 / size as f64;

    for y in 0..size {
        for x in 0..size {
            let mut sum = 0.0;

            for v in 0..size {
                for u in 0..size {
                    let cu = if u == 0 { 1.0 / 2.0_f64.sqrt() } else { 1.0 };
                    let cv = if v == 0 { 1.0 / 2.0_f64.sqrt() } else { 1.0 };

                    let cos_u = ((2 * x + 1) as f64 * u as f64 * PI / (2.0 * size as f64)).cos();
                    let cos_v = ((2 * y + 1) as f64 * v as f64 * PI / (2.0 * size as f64)).cos();

                    sum += cu * cv * input[v * size + u] * cos_u * cos_v;
                }
            }

            output[y * size + x] = scale * sum;
        }
    }

    output
}

/// Encode DCT coefficients to sparse representation
///
/// Returns only non-zero coefficients as (u, v, value) tuples
pub fn sparse_dct_encode(
    coefficients: &[i32],
    size: usize,
    threshold: f64,
) -> Vec<(u32, u32, f32)> {
    let threshold_i = (threshold * 1000.0) as i32;

    coefficients
        .iter()
        .enumerate()
        .filter(|(_, &c)| c.abs() > threshold_i)
        .map(|(i, &c)| {
            let u = (i % size) as u32;
            let v = (i / size) as u32;
            (u, v, c as f32)
        })
        .collect()
}

/// Decode sparse representation back to full coefficient matrix
pub fn sparse_dct_decode(sparse: &[(u32, u32, f32)], size: usize, default: i32) -> Vec<i32> {
    let mut coefficients = vec![default; size * size];

    for &(u, v, value) in sparse {
        let idx = v as usize * size + u as usize;
        if idx < coefficients.len() {
            coefficients[idx] = value as i32;
        }
    }

    coefficients
}

/// Process multiple blocks in parallel
pub fn dct2d_parallel(blocks: &[Vec<f64>], size: usize) -> Vec<Vec<f64>> {
    blocks.par_iter().map(|block| dct2d(block, size)).collect()
}

/// Process multiple blocks inverse in parallel
pub fn idct2d_parallel(blocks: &[Vec<f64>], size: usize) -> Vec<Vec<f64>> {
    blocks.par_iter().map(|block| idct2d(block, size)).collect()
}

/// Calculate energy compaction ratio (for quality measurement)
pub fn energy_compaction(original: &[f64], reconstructed: &[f64]) -> f64 {
    if original.len() != reconstructed.len() || original.is_empty() {
        return 0.0;
    }

    let original_energy: f64 = original.iter().map(|&x| x * x).sum();
    let error_energy: f64 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(&o, &r)| (o - r).powi(2))
        .sum();

    if original_energy > 0.0 {
        1.0 - (error_energy / original_energy)
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_idct_roundtrip() {
        let input: Vec<f64> = (0..64).map(|i| (i * 4) as f64).collect();
        let dct = dct2d(&input, 8);
        let output = idct2d(&dct, 8);

        for (i, (original, reconstructed)) in input.iter().zip(output.iter()).enumerate() {
            assert!(
                (original - reconstructed).abs() < 0.01,
                "Mismatch at {}: {} vs {}",
                i,
                original,
                reconstructed
            );
        }
    }

    #[test]
    fn test_sparse_encoding() {
        let coefficients = vec![100, 0, 0, 0, 50, 0, 0, 0, 0];
        let sparse = sparse_dct_encode(&coefficients, 3, 0.0);

        assert_eq!(sparse.len(), 2); // Only 2 non-zero values
        assert!(sparse.contains(&(0, 0, 100.0)));
        assert!(sparse.contains(&(1, 1, 50.0)));
    }

    #[test]
    fn test_sparse_decoding() {
        let sparse = vec![(0, 0, 100.0), (1, 1, 50.0)];
        let decoded = sparse_dct_decode(&sparse, 3, 0);

        assert_eq!(decoded[0], 100); // (0, 0)
        assert_eq!(decoded[4], 50); // (1, 1) = index 4
        assert_eq!(decoded[1], 0); // default
    }

    #[test]
    fn test_dct_transform_struct() {
        let transform = DctTransform::new(8).with_quality(50);

        let input: Vec<f64> = (0..64).map(|i| (i * 2) as f64).collect();
        let sparse = transform.encode_sparse(&input);
        let reconstructed = transform.decode_sparse(&sparse, 0.0);

        // Should have reasonable energy compaction
        let compaction = energy_compaction(&input, &reconstructed);
        assert!(compaction > 0.9, "Energy compaction: {}", compaction);
    }

    #[test]
    fn test_energy_compaction() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let perfect = vec![1.0, 2.0, 3.0, 4.0];
        let bad = vec![0.0, 0.0, 0.0, 0.0];

        assert!((energy_compaction(&original, &perfect) - 1.0).abs() < 0.001);
        assert!(energy_compaction(&original, &bad) < 0.001);
    }

    #[test]
    fn test_parallel_processing() {
        let blocks: Vec<Vec<f64>> = (0..16)
            .map(|_| (0..64).map(|i| i as f64).collect())
            .collect();

        let results = dct2d_parallel(&blocks, 8);
        assert_eq!(results.len(), 16);

        for result in &results {
            assert_eq!(result.len(), 64);
        }
    }
}
