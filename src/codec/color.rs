//! Color Palette Extraction
//!
//! High-performance color quantization using k-means clustering.
//! Optimized for video frame processing.

use crate::types::Color;
use rayon::prelude::*;

/// Color extractor configuration
#[derive(Debug, Clone)]
pub struct ColorExtractor {
    /// Number of dominant colors to extract
    pub num_colors: usize,
    /// Maximum iterations for k-means
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
    /// Sampling rate (1.0 = all pixels, 0.1 = 10% of pixels)
    pub sampling_rate: f64,
}

impl Default for ColorExtractor {
    fn default() -> Self {
        Self {
            num_colors: 5,
            max_iterations: 10,
            convergence_threshold: 1.0,
            sampling_rate: 0.1,
        }
    }
}

impl ColorExtractor {
    pub fn new(num_colors: usize) -> Self {
        Self {
            num_colors,
            ..Default::default()
        }
    }

    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.max_iterations = iterations;
        self
    }

    pub fn with_sampling_rate(mut self, rate: f64) -> Self {
        self.sampling_rate = rate.clamp(0.01, 1.0);
        self
    }

    /// Extract dominant colors from RGB image data
    pub fn extract(&self, pixels: &[u8]) -> Vec<Color> {
        extract_dominant_colors(pixels, self.num_colors, self.max_iterations, self.sampling_rate)
    }

    /// Extract with weights (how many pixels are closest to each color)
    pub fn extract_with_weights(&self, pixels: &[u8]) -> (Vec<Color>, Vec<f32>) {
        kmeans_palette(pixels, self.num_colors, self.max_iterations, self.sampling_rate)
    }
}

/// Extract dominant colors from RGB pixel data
///
/// # Arguments
/// * `pixels` - RGB pixel data (3 bytes per pixel)
/// * `k` - Number of colors to extract
/// * `max_iterations` - Maximum k-means iterations
/// * `sampling_rate` - Fraction of pixels to sample (for performance)
///
/// # Returns
/// Vector of dominant colors sorted by frequency
pub fn extract_dominant_colors(
    pixels: &[u8],
    k: usize,
    max_iterations: usize,
    sampling_rate: f64,
) -> Vec<Color> {
    let (colors, _) = kmeans_palette(pixels, k, max_iterations, sampling_rate);
    colors
}

/// K-means color palette extraction with weights
pub fn kmeans_palette(
    pixels: &[u8],
    k: usize,
    max_iterations: usize,
    sampling_rate: f64,
) -> (Vec<Color>, Vec<f32>) {
    if pixels.len() < 3 {
        return (vec![Color::black()], vec![1.0]);
    }

    // Sample pixels for performance
    let sampled = sample_pixels(pixels, sampling_rate);
    if sampled.is_empty() {
        return (vec![Color::black()], vec![1.0]);
    }

    // Initialize centroids using k-means++
    let mut centroids = kmeans_pp_init(&sampled, k);

    // Run k-means iterations
    let mut assignments = vec![0usize; sampled.len()];

    for _ in 0..max_iterations {
        // Assign points to nearest centroid
        let changed = assign_to_centroids(&sampled, &centroids, &mut assignments);

        // Update centroids
        centroids = update_centroids(&sampled, &assignments, k);

        // Check for convergence
        if !changed {
            break;
        }
    }

    // Calculate weights (fraction of pixels assigned to each centroid)
    let mut counts = vec![0u32; k];
    for &a in &assignments {
        if a < k {
            counts[a] += 1;
        }
    }

    let total = sampled.len() as f32;
    let weights: Vec<f32> = counts.iter().map(|&c| c as f32 / total).collect();

    // Convert centroids to Colors and sort by weight
    let mut color_weights: Vec<(Color, f32)> = centroids
        .into_iter()
        .zip(weights)
        .filter(|(_, w)| *w > 0.0)
        .collect();

    color_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let colors: Vec<Color> = color_weights.iter().map(|(c, _)| *c).collect();
    let sorted_weights: Vec<f32> = color_weights.iter().map(|(_, w)| *w).collect();

    if colors.is_empty() {
        (vec![Color::black()], vec![1.0])
    } else {
        (colors, sorted_weights)
    }
}

/// Sample pixels from the image
fn sample_pixels(pixels: &[u8], rate: f64) -> Vec<[u8; 3]> {
    let num_pixels = pixels.len() / 3;
    let step = (1.0 / rate).max(1.0) as usize;

    (0..num_pixels)
        .step_by(step)
        .map(|i| {
            let base = i * 3;
            [pixels[base], pixels[base + 1], pixels[base + 2]]
        })
        .collect()
}

/// K-means++ initialization
fn kmeans_pp_init(pixels: &[[u8; 3]], k: usize) -> Vec<Color> {
    if pixels.is_empty() || k == 0 {
        return vec![];
    }

    let mut centroids = Vec::with_capacity(k);

    // First centroid: random pixel (we use middle for determinism)
    let first_idx = pixels.len() / 2;
    centroids.push(Color::from_array(pixels[first_idx]));

    // Remaining centroids: probability proportional to distance squared
    for _ in 1..k {
        // Calculate distances to nearest centroid (parallel)
        let distances: Vec<f64> = pixels
            .par_iter()
            .map(|p| {
                centroids
                    .iter()
                    .map(|c| color_distance_sq(p, c))
                    .fold(f64::MAX, f64::min)
            })
            .collect();

        // Find point with maximum distance (greedy approximation of k-means++)
        let max_idx = distances
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        centroids.push(Color::from_array(pixels[max_idx]));
    }

    centroids
}

/// Assign pixels to nearest centroid
fn assign_to_centroids(
    pixels: &[[u8; 3]],
    centroids: &[Color],
    assignments: &mut [usize],
) -> bool {
    let new_assignments: Vec<usize> = pixels
        .par_iter()
        .map(|p| {
            centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    color_distance_sq(p, a)
                        .partial_cmp(&color_distance_sq(p, b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .collect();

    let changed = new_assignments != assignments;
    assignments.copy_from_slice(&new_assignments);
    changed
}

/// Update centroids based on assignments
fn update_centroids(pixels: &[[u8; 3]], assignments: &[usize], k: usize) -> Vec<Color> {
    // Parallel accumulation
    let (sums, counts): (Vec<[u64; 3]>, Vec<u32>) = (0..k)
        .into_par_iter()
        .map(|cluster| {
            let mut sum = [0u64; 3];
            let mut count = 0u32;

            for (pixel, &assignment) in pixels.iter().zip(assignments.iter()) {
                if assignment == cluster {
                    sum[0] += pixel[0] as u64;
                    sum[1] += pixel[1] as u64;
                    sum[2] += pixel[2] as u64;
                    count += 1;
                }
            }

            (sum, count)
        })
        .unzip();

    sums.into_iter()
        .zip(counts)
        .map(|(sum, count)| {
            if count > 0 {
                Color::new(
                    (sum[0] / count as u64) as u8,
                    (sum[1] / count as u64) as u8,
                    (sum[2] / count as u64) as u8,
                )
            } else {
                Color::black()
            }
        })
        .collect()
}

/// Calculate squared color distance (Euclidean in RGB space)
#[inline]
fn color_distance_sq(pixel: &[u8; 3], color: &Color) -> f64 {
    let dr = pixel[0] as f64 - color.r as f64;
    let dg = pixel[1] as f64 - color.g as f64;
    let db = pixel[2] as f64 - color.b as f64;
    dr * dr + dg * dg + db * db
}

/// Reduce color palette using median cut algorithm (alternative to k-means)
pub fn median_cut_palette(pixels: &[u8], num_colors: usize) -> Vec<Color> {
    if pixels.len() < 3 || num_colors == 0 {
        return vec![Color::black()];
    }

    // Build histogram of unique colors
    let mut colors: Vec<[u8; 3]> = (0..pixels.len() / 3)
        .map(|i| [pixels[i * 3], pixels[i * 3 + 1], pixels[i * 3 + 2]])
        .collect();

    if colors.is_empty() {
        return vec![Color::black()];
    }

    // Recursively split color space
    fn median_cut(colors: &mut [[u8; 3]], depth: usize, max_depth: usize, result: &mut Vec<Color>) {
        if colors.is_empty() {
            return;
        }

        if depth >= max_depth || colors.len() <= 1 {
            // Average the colors in this bucket
            let (sum_r, sum_g, sum_b) = colors.iter().fold((0u64, 0u64, 0u64), |acc, c| {
                (acc.0 + c[0] as u64, acc.1 + c[1] as u64, acc.2 + c[2] as u64)
            });
            let n = colors.len() as u64;
            result.push(Color::new(
                (sum_r / n) as u8,
                (sum_g / n) as u8,
                (sum_b / n) as u8,
            ));
            return;
        }

        // Find channel with greatest range
        let (min_r, max_r) = colors.iter().fold((255u8, 0u8), |acc, c| (acc.0.min(c[0]), acc.1.max(c[0])));
        let (min_g, max_g) = colors.iter().fold((255u8, 0u8), |acc, c| (acc.0.min(c[1]), acc.1.max(c[1])));
        let (min_b, max_b) = colors.iter().fold((255u8, 0u8), |acc, c| (acc.0.min(c[2]), acc.1.max(c[2])));

        let range_r = max_r - min_r;
        let range_g = max_g - min_g;
        let range_b = max_b - min_b;

        // Sort by channel with greatest range
        if range_r >= range_g && range_r >= range_b {
            colors.sort_by_key(|c| c[0]);
        } else if range_g >= range_b {
            colors.sort_by_key(|c| c[1]);
        } else {
            colors.sort_by_key(|c| c[2]);
        }

        // Split at median
        let mid = colors.len() / 2;
        let (left, right) = colors.split_at_mut(mid);

        median_cut(left, depth + 1, max_depth, result);
        median_cut(right, depth + 1, max_depth, result);
    }

    let max_depth = (num_colors as f64).log2().ceil() as usize;
    let mut result = Vec::with_capacity(num_colors);
    median_cut(&mut colors, 0, max_depth, &mut result);

    result.truncate(num_colors);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_single_color() {
        // All red pixels
        let pixels: Vec<u8> = (0..300).flat_map(|_| [255, 0, 0]).collect();
        let colors = extract_dominant_colors(&pixels, 3, 10, 1.0);

        assert!(!colors.is_empty());
        // Should have at least one color close to red
        assert!(colors.iter().any(|c| c.r > 200 && c.g < 50 && c.b < 50));
    }

    #[test]
    fn test_extract_multiple_colors() {
        // Half red, half blue
        let mut pixels = Vec::new();
        for _ in 0..50 {
            pixels.extend_from_slice(&[255, 0, 0]); // Red
        }
        for _ in 0..50 {
            pixels.extend_from_slice(&[0, 0, 255]); // Blue
        }

        let colors = extract_dominant_colors(&pixels, 2, 10, 1.0);

        assert_eq!(colors.len(), 2);
        // Should have both red and blue
        let has_red = colors.iter().any(|c| c.r > 200 && c.b < 50);
        let has_blue = colors.iter().any(|c| c.b > 200 && c.r < 50);
        assert!(has_red && has_blue);
    }

    #[test]
    fn test_kmeans_with_weights() {
        // 75% red, 25% blue
        let mut pixels = Vec::new();
        for _ in 0..75 {
            pixels.extend_from_slice(&[255, 0, 0]);
        }
        for _ in 0..25 {
            pixels.extend_from_slice(&[0, 0, 255]);
        }

        let (colors, weights) = kmeans_palette(&pixels, 2, 10, 1.0);

        assert_eq!(colors.len(), 2);
        assert_eq!(weights.len(), 2);

        // First color (highest weight) should be red
        assert!(colors[0].r > 200);
        assert!(weights[0] > 0.5);
    }

    #[test]
    fn test_sampling_rate() {
        let pixels: Vec<u8> = (0..30000).map(|i| (i % 256) as u8).collect();

        // Should work with different sampling rates
        let colors_full = extract_dominant_colors(&pixels, 5, 10, 1.0);
        let colors_sampled = extract_dominant_colors(&pixels, 5, 10, 0.1);

        assert!(!colors_full.is_empty());
        assert!(!colors_sampled.is_empty());
    }

    #[test]
    fn test_median_cut() {
        let mut pixels = Vec::new();
        for _ in 0..50 {
            pixels.extend_from_slice(&[255, 0, 0]);
            pixels.extend_from_slice(&[0, 255, 0]);
        }

        let colors = median_cut_palette(&pixels, 4);

        assert!(!colors.is_empty());
        assert!(colors.len() <= 4);
    }

    #[test]
    fn test_color_extractor_struct() {
        let extractor = ColorExtractor::new(3)
            .with_iterations(5)
            .with_sampling_rate(0.5);

        assert_eq!(extractor.num_colors, 3);
        assert_eq!(extractor.max_iterations, 5);
        assert!((extractor.sampling_rate - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_empty_input() {
        let colors = extract_dominant_colors(&[], 5, 10, 1.0);
        assert_eq!(colors.len(), 1);
        assert_eq!(colors[0], Color::black());
    }
}
