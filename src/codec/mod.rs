//! ASP Codec Module
//!
//! High-performance video encoding/decoding components.
//!
//! This module provides:
//! - Motion estimation (parallel, SIMD-optimized)
//! - DCT transform
//! - Color palette extraction
//! - ROI detection

pub mod motion;
pub mod dct;
pub mod color;
pub mod roi;

pub use motion::{estimate_motion, estimate_motion_parallel, MotionEstimator, SearchAlgorithm};
pub use dct::{dct2d, idct2d, sparse_dct_encode, sparse_dct_decode, DctTransform};
pub use color::{extract_dominant_colors, kmeans_palette, ColorExtractor};
pub use roi::{detect_rois, RoiDetector, RoiConfig};
