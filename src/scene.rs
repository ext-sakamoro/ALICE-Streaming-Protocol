//! SDF Scene Channel for ALICE Hybrid Streaming
//!
//! Enables SDF-based procedural background transmission:
//!
//! ```text
//! Traditional:  Background pixels (70-80% of frame) → H.265 → 5-10 Mbps
//! ALICE Hybrid: Background SDF description            → ASP   → 2-10 KB
//! ```
//!
//! The SDF scene description replaces pixel-based backgrounds with a compact
//! procedural representation. Only the person/foreground is encoded as video.

use serde::{Deserialize, Serialize};

/// SDF Scene descriptor embedded in I-Packets.
///
/// Contains a complete scene description as ASDF binary (from ALICE-SDF).
/// Typical size: 2-10 KB for complex 3D scenes vs. 500KB+ for pixel data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdfSceneDescriptor {
    /// ASDF binary blob (ALICE-SDF serialized scene tree)
    /// Format: [Magic "ASDF" 4B][Version 2B][Flags 2B][NodeCount 4B][CRC 4B][BincodeBody...]
    pub asdf_data: Vec<u8>,
    /// Scene bounds: (min_x, min_y, min_z, max_x, max_y, max_z)
    pub bounds: [f32; 6],
    /// Render resolution hint (0 = use frame resolution)
    pub render_resolution: u32,
    /// Background color (fallback if SDF rendering not supported)
    pub fallback_color: [u8; 3],
    /// Scene version (monotonically increasing, for delta detection)
    pub scene_version: u32,
    /// Optional scene name/identifier
    pub scene_name: Option<String>,
    /// Optional classification labels from edge ML (label_id → label_name)
    pub classification_labels: Option<Vec<(u8, String)>>,
}

impl SdfSceneDescriptor {
    /// Create a new SDF scene descriptor from ASDF binary data
    pub fn new(asdf_data: Vec<u8>) -> Self {
        Self {
            asdf_data,
            bounds: [-10.0, -10.0, -10.0, 10.0, 10.0, 10.0],
            render_resolution: 0,
            fallback_color: [0, 0, 0],
            scene_version: 1,
            scene_name: None,
            classification_labels: None,
        }
    }

    /// Set scene bounds
    pub fn with_bounds(mut self, min: [f32; 3], max: [f32; 3]) -> Self {
        self.bounds = [min[0], min[1], min[2], max[0], max[1], max[2]];
        self
    }

    /// Set fallback color
    pub fn with_fallback_color(mut self, r: u8, g: u8, b: u8) -> Self {
        self.fallback_color = [r, g, b];
        self
    }

    /// Set render resolution hint
    pub fn with_render_resolution(mut self, res: u32) -> Self {
        self.render_resolution = res;
        self
    }

    /// Set scene name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.scene_name = Some(name.into());
        self
    }

    /// Get ASDF data size in bytes
    pub fn asdf_size(&self) -> usize {
        self.asdf_data.len()
    }

    /// Validate ASDF magic bytes ("ASDF")
    pub fn is_valid_asdf(&self) -> bool {
        self.asdf_data.len() >= 16
            && self.asdf_data[0] == b'A'
            && self.asdf_data[1] == b'S'
            && self.asdf_data[2] == b'D'
            && self.asdf_data[3] == b'F'
    }
}

/// SDF Scene delta for D-Packets.
///
/// Instead of re-sending the entire scene, only send what changed:
/// - Animation parameter updates (camera, lighting, object transforms)
/// - Node-level patches (add/remove/modify individual SDF nodes)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdfSceneDelta {
    /// Reference scene version (from I-Packet's scene_version)
    pub ref_scene_version: u32,
    /// New scene version after applying this delta
    pub new_scene_version: u32,
    /// Delta type
    pub delta_type: SdfDeltaType,
    /// Delta payload (interpretation depends on delta_type)
    pub delta_data: Vec<u8>,
}

/// Types of SDF scene deltas
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum SdfDeltaType {
    /// Full scene replacement (fallback, equivalent to I-Packet scene)
    FullReplace = 0x00,
    /// Animation parameter update (camera position, rotation, etc.)
    AnimationUpdate = 0x01,
    /// Node transform update (translate/rotate/scale specific nodes)
    NodeTransform = 0x02,
    /// Node material update (color, texture changes)
    MaterialUpdate = 0x03,
    /// Node add/remove (structural scene changes)
    StructuralChange = 0x04,
    /// Lighting update
    LightingUpdate = 0x05,
    /// SVO chunk delta (Sparse Voxel Octree incremental update from edge devices)
    SvoChunkDelta = 0x06,
}

impl SdfSceneDelta {
    /// Create an animation update delta
    pub fn animation_update(ref_version: u32, data: Vec<u8>) -> Self {
        Self {
            ref_scene_version: ref_version,
            new_scene_version: ref_version + 1,
            delta_type: SdfDeltaType::AnimationUpdate,
            delta_data: data,
        }
    }

    /// Create a node transform delta
    pub fn node_transform(ref_version: u32, data: Vec<u8>) -> Self {
        Self {
            ref_scene_version: ref_version,
            new_scene_version: ref_version + 1,
            delta_type: SdfDeltaType::NodeTransform,
            delta_data: data,
        }
    }

    /// Create a full scene replacement delta
    pub fn full_replace(ref_version: u32, asdf_data: Vec<u8>) -> Self {
        Self {
            ref_scene_version: ref_version,
            new_scene_version: ref_version + 1,
            delta_type: SdfDeltaType::FullReplace,
            delta_data: asdf_data,
        }
    }

    /// Create an SVO chunk delta (from edge device 3D scanner)
    pub fn svo_chunk_delta(ref_version: u32, chunk_data: Vec<u8>) -> Self {
        Self {
            ref_scene_version: ref_version,
            new_scene_version: ref_version + 1,
            delta_type: SdfDeltaType::SvoChunkDelta,
            delta_data: chunk_data,
        }
    }

    /// Get delta payload size in bytes
    pub fn delta_size(&self) -> usize {
        self.delta_data.len()
    }
}

/// Person mask descriptor for hybrid streaming.
///
/// Defines the foreground region that contains the person,
/// separating it from the SDF-rendered background.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonMask {
    /// Bounding box of the person region [x, y, width, height]
    pub bbox: [u32; 4],
    /// RLE-compressed binary mask within the bounding box
    /// Format: [run_length: u16, value: u8 (0 or 1)]...
    pub rle_mask: Vec<u8>,
    /// Mask confidence (0.0 - 1.0)
    pub confidence: f32,
    /// Number of foreground pixels
    pub foreground_pixels: u32,
    /// Total pixels in bounding box
    pub total_pixels: u32,
}

impl PersonMask {
    /// Create a new person mask
    pub fn new(bbox: [u32; 4], rle_mask: Vec<u8>) -> Self {
        Self {
            bbox,
            rle_mask,
            confidence: 1.0,
            foreground_pixels: 0,
            total_pixels: bbox[2] * bbox[3],
        }
    }

    /// Foreground coverage ratio (0.0 - 1.0)
    pub fn coverage(&self) -> f32 {
        if self.total_pixels == 0 {
            return 0.0;
        }
        self.foreground_pixels as f32 / self.total_pixels as f32
    }

    /// Estimated bandwidth savings vs. full-frame encoding
    ///
    /// Returns a ratio: 1.0 = no savings, 0.2 = 80% savings
    pub fn bandwidth_ratio(&self, frame_width: u32, frame_height: u32) -> f32 {
        let frame_pixels = frame_width as f64 * frame_height as f64;
        if frame_pixels == 0.0 {
            return 1.0;
        }
        self.foreground_pixels as f32 / frame_pixels as f32
    }

    /// RLE mask size in bytes
    pub fn mask_size(&self) -> usize {
        self.rle_mask.len()
    }
}

/// Encode a binary mask (1 = foreground, 0 = background) to RLE — optimized.
///
/// RLE format: pairs of (run_length_u16_le, value_u8)
/// Each pair = 3 bytes. Typical person mask: 200-500 bytes.
///
/// Optimization: scan-forward loop finds run boundaries in bulk
/// (compiler auto-vectorizes the inner equality scan).
pub fn rle_encode_mask(mask: &[u8], width: u32, height: u32) -> Vec<u8> {
    let total = (width * height) as usize;
    if total == 0 || mask.is_empty() {
        return Vec::new();
    }
    let n = total.min(mask.len());
    let mut rle = Vec::with_capacity(256);
    let mut pos = 0;

    while pos < n {
        let val = mask[pos] & 1;
        let start = pos;
        // Scan forward for same value — inner loop auto-vectorizes
        while pos < n && (mask[pos] & 1) == val && (pos - start) < u16::MAX as usize {
            pos += 1;
        }
        let run_len = (pos - start) as u16;
        let bytes = run_len.to_le_bytes();
        rle.push(bytes[0]);
        rle.push(bytes[1]);
        rle.push(val);
    }
    rle
}

/// Decode RLE mask back to binary mask — optimized.
///
/// Pre-allocates zero-filled buffer, only fills non-zero runs.
/// Since masks are mostly background (0), this skips ~70-80% of writes.
pub fn rle_decode_mask(rle: &[u8], width: u32, height: u32) -> Vec<u8> {
    let total = (width * height) as usize;
    // Pre-allocate as zeros (memset, SIMD-optimized by allocator)
    let mut mask = vec![0u8; total];
    let mut write_pos = 0;

    let mut i = 0;
    while i + 2 < rle.len() && write_pos < total {
        let run_len = u16::from_le_bytes([rle[i], rle[i + 1]]) as usize;
        let val = rle[i + 2];
        let count = run_len.min(total - write_pos);
        if val != 0 {
            // Only fill non-zero runs (mask is already zeroed)
            // .fill() compiles to optimized memset
            mask[write_pos..write_pos + count].fill(val);
        }
        write_pos += count;
        i += 3;
    }
    mask.truncate(total);
    mask
}

/// Bandwidth comparison statistics for hybrid vs. traditional streaming
#[derive(Debug, Clone, Default)]
pub struct HybridBandwidthStats {
    /// SDF scene data size (bytes, typically 2-10 KB)
    pub sdf_scene_bytes: usize,
    /// SDF delta data size (bytes per delta frame)
    pub sdf_delta_bytes: usize,
    /// Person mask size (bytes, RLE compressed)
    pub person_mask_bytes: usize,
    /// Person video data size (bytes, wavelet encoded)
    pub person_video_bytes: usize,
    /// Audio data size (bytes, LPC/spectral encoded)
    pub audio_bytes: usize,
    /// Total hybrid stream size (bytes)
    pub hybrid_total_bytes: usize,
    /// Estimated traditional full-frame size (bytes)
    pub traditional_total_bytes: usize,
    /// Frame dimensions
    pub frame_width: u32,
    pub frame_height: u32,
    /// Number of frames
    pub frame_count: u32,
}

impl HybridBandwidthStats {
    /// Calculate bandwidth savings ratio
    ///
    /// Returns: (savings_percent, compression_ratio)
    /// Example: (92.5, 13.3) means 92.5% savings, 13.3x compression
    pub fn savings(&self) -> (f64, f64) {
        if self.traditional_total_bytes == 0 {
            return (0.0, 1.0);
        }
        let ratio = self.traditional_total_bytes as f64 / self.hybrid_total_bytes.max(1) as f64;
        let savings =
            (1.0 - self.hybrid_total_bytes as f64 / self.traditional_total_bytes as f64) * 100.0;
        (savings, ratio)
    }

    /// Format as human-readable report
    pub fn report(&self) -> String {
        let (savings, ratio) = self.savings();
        format!(
            "ALICE Hybrid Bandwidth Report\n\
             ==============================\n\
             Frame: {}x{} × {} frames\n\
             \n\
             SDF Scene:      {:>8} bytes ({:.1} KB)\n\
             SDF Deltas:     {:>8} bytes ({:.1} KB)\n\
             Person Mask:    {:>8} bytes ({:.1} KB)\n\
             Person Video:   {:>8} bytes ({:.1} KB)\n\
             Audio:          {:>8} bytes ({:.1} KB)\n\
             ─────────────────────────────\n\
             Hybrid Total:   {:>8} bytes ({:.1} KB)\n\
             Traditional:    {:>8} bytes ({:.1} KB)\n\
             \n\
             Savings: {:.1}% (compression ratio: {:.1}x)",
            self.frame_width,
            self.frame_height,
            self.frame_count,
            self.sdf_scene_bytes,
            self.sdf_scene_bytes as f64 / 1024.0,
            self.sdf_delta_bytes,
            self.sdf_delta_bytes as f64 / 1024.0,
            self.person_mask_bytes,
            self.person_mask_bytes as f64 / 1024.0,
            self.person_video_bytes,
            self.person_video_bytes as f64 / 1024.0,
            self.audio_bytes,
            self.audio_bytes as f64 / 1024.0,
            self.hybrid_total_bytes,
            self.hybrid_total_bytes as f64 / 1024.0,
            self.traditional_total_bytes,
            self.traditional_total_bytes as f64 / 1024.0,
            savings,
            ratio,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdf_scene_descriptor() {
        // Simulate ASDF binary data with magic header
        let mut asdf_data = vec![b'A', b'S', b'D', b'F'];
        asdf_data.extend_from_slice(&[0x01, 0x00]); // version
        asdf_data.extend_from_slice(&[0x00, 0x00]); // flags
        asdf_data.extend_from_slice(&[0x05, 0x00, 0x00, 0x00]); // node count
        asdf_data.extend_from_slice(&[0x00; 4]); // CRC placeholder
        asdf_data.extend_from_slice(&[0xDE, 0xAD]); // body placeholder

        let scene = SdfSceneDescriptor::new(asdf_data.clone())
            .with_bounds([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0])
            .with_fallback_color(30, 30, 50)
            .with_name("test_scene");

        assert!(scene.is_valid_asdf());
        assert_eq!(scene.asdf_size(), asdf_data.len());
        assert_eq!(scene.scene_version, 1);
        assert_eq!(scene.scene_name.as_deref(), Some("test_scene"));
    }

    #[test]
    fn test_sdf_scene_delta() {
        let delta = SdfSceneDelta::animation_update(1, vec![0x01, 0x02, 0x03]);
        assert_eq!(delta.ref_scene_version, 1);
        assert_eq!(delta.new_scene_version, 2);
        assert_eq!(delta.delta_type, SdfDeltaType::AnimationUpdate);
        assert_eq!(delta.delta_size(), 3);
    }

    #[test]
    fn test_rle_encode_decode_roundtrip() {
        // Create a simple mask: 10 background, 20 foreground, 10 background
        let width = 10u32;
        let height = 4u32;
        let mut mask = vec![0u8; 40];
        for i in 10..30 {
            mask[i] = 1;
        }

        let rle = rle_encode_mask(&mask, width, height);
        let decoded = rle_decode_mask(&rle, width, height);

        assert_eq!(mask, decoded);
        // RLE should be much smaller: 3 runs × 3 bytes = 9 bytes vs 40 bytes
        assert!(rle.len() < mask.len());
        assert_eq!(rle.len(), 9); // 3 runs × 3 bytes each
    }

    #[test]
    fn test_rle_all_zeros() {
        let mask = vec![0u8; 100];
        let rle = rle_encode_mask(&mask, 10, 10);
        let decoded = rle_decode_mask(&rle, 10, 10);
        assert_eq!(mask, decoded);
        assert_eq!(rle.len(), 3); // single run
    }

    #[test]
    fn test_rle_alternating() {
        let mask: Vec<u8> = (0..20).map(|i| i % 2).collect();
        let rle = rle_encode_mask(&mask, 20, 1);
        let decoded = rle_decode_mask(&rle, 20, 1);
        assert_eq!(mask, decoded);
    }

    #[test]
    fn test_person_mask() {
        let mask = PersonMask {
            bbox: [100, 50, 200, 400],
            rle_mask: vec![0; 100],
            confidence: 0.95,
            foreground_pixels: 50000,
            total_pixels: 80000,
        };
        assert!((mask.coverage() - 0.625).abs() < 0.001);

        // 1080p frame, person is ~50000 pixels out of 2M
        let ratio = mask.bandwidth_ratio(1920, 1080);
        assert!(ratio < 0.05); // person < 5% of frame → huge savings
    }

    #[test]
    fn test_hybrid_bandwidth_stats() {
        let stats = HybridBandwidthStats {
            sdf_scene_bytes: 5_000,     // 5 KB SDF scene
            sdf_delta_bytes: 500,       // 500 bytes deltas
            person_mask_bytes: 300,     // 300 bytes mask
            person_video_bytes: 50_000, // 50 KB person video
            audio_bytes: 0,
            hybrid_total_bytes: 55_800,       // total
            traditional_total_bytes: 500_000, // 500 KB traditional
            frame_width: 1920,
            frame_height: 1080,
            frame_count: 30,
        };

        let (savings, ratio) = stats.savings();
        assert!(savings > 80.0); // >80% savings
        assert!(ratio > 5.0); // >5x compression
    }
}
