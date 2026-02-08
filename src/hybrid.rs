//! ALICE Hybrid Streaming Pipeline
//!
//! Integrates ALICE-SDF (background) + ALICE-Codec (person) + ASP (transport).
//!
//! ```text
//! ┌─────────────── Transmitter ───────────────┐
//! │                                            │
//! │  Camera → Segmentation → Person Mask       │
//! │              ↓               ↓             │
//! │  SDF Scene (2-10KB)   Person Crop          │
//! │       ↓                    ↓               │
//! │  I-Packet.sdf_scene   Wavelet Encode       │
//! │  D-Packet.sdf_delta   C-Packet (face ROI)  │
//! │              ↓               ↓             │
//! │          ASP Multiplexer → Network         │
//! └────────────────────────────────────────────┘
//!
//! ┌─────────────── Receiver ──────────────────┐
//! │                                            │
//! │  Network → ASP Demultiplexer               │
//! │              ↓               ↓             │
//! │  SDF Scene              Person Stream      │
//! │       ↓                    ↓               │
//! │  SDF Render (GPU)     Wavelet Decode       │
//! │       ↓                    ↓               │
//! │  Background Frame     Person Pixels + Mask │
//! │              ↓               ↓             │
//! │          Compositor → Display              │
//! └────────────────────────────────────────────┘
//! ```
//!
//! # Bandwidth Model
//!
//! | Component | Traditional (H.265) | ALICE Hybrid |
//! |-----------|--------------------:|-------------:|
//! | Background (70-80%) | 3.5-8 Mbps | **2-10 KB** (SDF, once) |
//! | Person (20-30%) | 1-2 Mbps | **0.5-2 Mbps** (wavelet) |
//! | Mask | N/A | **~300 bytes** (RLE) |
//! | **Total** | **5-10 Mbps** | **~0.5-2 Mbps** |

use crate::scene::{
    SdfSceneDescriptor, SdfSceneDelta, PersonMask,
    HybridBandwidthStats, rle_encode_mask,
};
use serde::{Deserialize, Serialize};

/// Hybrid frame: combines SDF background + person video + audio data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridFrame {
    /// Frame sequence number
    pub sequence: u32,
    /// Frame timestamp (ms)
    pub timestamp_ms: u64,
    /// Frame dimensions
    pub width: u32,
    pub height: u32,
    /// SDF background scene (present in I-frames, None in D-frames)
    pub sdf_scene: Option<SdfSceneDescriptor>,
    /// SDF delta (present in D-frames for animated scenes)
    pub sdf_delta: Option<SdfSceneDelta>,
    /// Person mask (bounding box + RLE mask)
    pub person_mask: Option<PersonMask>,
    /// Person video data (wavelet-encoded, only the person region)
    pub person_video_data: Vec<u8>,
    /// Audio data (encoded voice/audio parameters, empty if no audio)
    pub audio_data: Vec<u8>,
    /// Whether this is a keyframe
    pub is_keyframe: bool,
}

/// Hybrid transmitter: packages SDF scene + person video into ASP packets
#[derive(Debug)]
pub struct HybridTransmitter {
    /// Current sequence number
    sequence: u32,
    /// Current scene version
    scene_version: u32,
    /// Bandwidth statistics accumulator
    stats: HybridBandwidthStats,
}

impl HybridTransmitter {
    /// Create a new hybrid transmitter
    pub fn new() -> Self {
        Self {
            sequence: 0,
            scene_version: 0,
            stats: HybridBandwidthStats::default(),
        }
    }

    /// Create a keyframe I-Packet with SDF scene + initial person data + optional audio.
    ///
    /// This is sent at the start of a stream or on scene changes.
    /// Contains the full SDF scene description + first person frame + audio.
    pub fn create_keyframe(
        &mut self,
        width: u32,
        height: u32,
        _fps: f32,
        sdf_scene: SdfSceneDescriptor,
        person_mask: Option<PersonMask>,
        person_video: Vec<u8>,
    ) -> HybridFrame {
        self.create_keyframe_av(width, height, _fps, sdf_scene, person_mask, person_video, Vec::new())
    }

    /// Create a keyframe with audio/video.
    ///
    /// Same as `create_keyframe` but also carries audio data.
    pub fn create_keyframe_av(
        &mut self,
        width: u32,
        height: u32,
        _fps: f32,
        sdf_scene: SdfSceneDescriptor,
        person_mask: Option<PersonMask>,
        person_video: Vec<u8>,
        audio_data: Vec<u8>,
    ) -> HybridFrame {
        self.sequence += 1;
        self.scene_version = sdf_scene.scene_version;

        // Update stats
        self.stats.sdf_scene_bytes += sdf_scene.asdf_size();
        if let Some(ref mask) = person_mask {
            self.stats.person_mask_bytes += mask.mask_size();
        }
        self.stats.person_video_bytes += person_video.len();
        self.stats.audio_bytes += audio_data.len();
        self.stats.frame_width = width;
        self.stats.frame_height = height;
        self.stats.frame_count += 1;

        let hybrid_size = sdf_scene.asdf_size()
            + person_mask.as_ref().map_or(0, |m| m.mask_size())
            + person_video.len()
            + audio_data.len();
        self.stats.hybrid_total_bytes += hybrid_size;
        // Estimate traditional: raw frame ≈ width × height × 3 / 20 (H.265 ~20:1)
        self.stats.traditional_total_bytes += (width * height * 3 / 20) as usize;

        HybridFrame {
            sequence: self.sequence,
            timestamp_ms: 0,
            width,
            height,
            sdf_scene: Some(sdf_scene),
            sdf_delta: None,
            person_mask,
            person_video_data: person_video,
            audio_data,
            is_keyframe: true,
        }
    }

    /// Create a delta D-Packet with optional SDF delta + person update.
    ///
    /// This is the common case: background unchanged (0 bytes for SDF),
    /// only person video data is transmitted.
    pub fn create_delta_frame(
        &mut self,
        sdf_delta: Option<SdfSceneDelta>,
        person_mask: Option<PersonMask>,
        person_video: Vec<u8>,
        timestamp_ms: u64,
    ) -> HybridFrame {
        self.create_delta_frame_av(sdf_delta, person_mask, person_video, Vec::new(), timestamp_ms)
    }

    /// Create a delta frame with audio/video.
    ///
    /// Same as `create_delta_frame` but also carries audio data.
    pub fn create_delta_frame_av(
        &mut self,
        sdf_delta: Option<SdfSceneDelta>,
        person_mask: Option<PersonMask>,
        person_video: Vec<u8>,
        audio_data: Vec<u8>,
        timestamp_ms: u64,
    ) -> HybridFrame {
        self.sequence += 1;

        if let Some(ref delta) = sdf_delta {
            self.stats.sdf_delta_bytes += delta.delta_size();
            self.scene_version = delta.new_scene_version;
        }
        if let Some(ref mask) = person_mask {
            self.stats.person_mask_bytes += mask.mask_size();
        }
        self.stats.person_video_bytes += person_video.len();
        self.stats.audio_bytes += audio_data.len();
        self.stats.frame_count += 1;

        let hybrid_size = sdf_delta.as_ref().map_or(0, |d| d.delta_size())
            + person_mask.as_ref().map_or(0, |m| m.mask_size())
            + person_video.len()
            + audio_data.len();
        self.stats.hybrid_total_bytes += hybrid_size;
        self.stats.traditional_total_bytes +=
            (self.stats.frame_width * self.stats.frame_height * 3 / 20) as usize;

        HybridFrame {
            sequence: self.sequence,
            timestamp_ms,
            width: self.stats.frame_width,
            height: self.stats.frame_height,
            sdf_scene: None,
            sdf_delta,
            person_mask,
            person_video_data: person_video,
            audio_data,
            is_keyframe: false,
        }
    }

    /// Get current bandwidth statistics
    pub fn stats(&self) -> &HybridBandwidthStats {
        &self.stats
    }

    /// Get current sequence number
    pub fn sequence(&self) -> u32 {
        self.sequence
    }

    /// Get current scene version
    pub fn scene_version(&self) -> u32 {
        self.scene_version
    }
}

impl Default for HybridTransmitter {
    fn default() -> Self {
        Self::new()
    }
}

/// Hybrid receiver: demultiplexes ASP packets into SDF scene + person video
#[derive(Debug)]
pub struct HybridReceiver {
    /// Last received SDF scene
    current_scene: Option<SdfSceneDescriptor>,
    /// Scene version tracker
    scene_version: u32,
    /// Last received person mask
    current_mask: Option<PersonMask>,
    /// Frame dimensions
    width: u32,
    height: u32,
    /// Frames received
    frames_received: u32,
}

impl HybridReceiver {
    /// Create a new hybrid receiver
    pub fn new() -> Self {
        Self {
            current_scene: None,
            scene_version: 0,
            current_mask: None,
            width: 0,
            height: 0,
            frames_received: 0,
        }
    }

    /// Process an incoming hybrid frame.
    ///
    /// Returns a `CompositeInstruction` describing how to assemble the final frame.
    pub fn process_frame(&mut self, frame: &HybridFrame) -> CompositeInstruction {
        self.frames_received += 1;

        if frame.is_keyframe {
            if let Some(ref scene) = frame.sdf_scene {
                self.current_scene = Some(scene.clone());
                self.scene_version = scene.scene_version;
            }
            self.width = frame.width;
            self.height = frame.height;
        }

        if let Some(ref mask) = frame.person_mask {
            self.current_mask = Some(mask.clone());
        }

        CompositeInstruction {
            render_sdf_background: self.current_scene.is_some(),
            sdf_scene_version: self.scene_version,
            person_bbox: self.current_mask.as_ref().map(|m| m.bbox),
            person_video_size: frame.person_video_data.len(),
            frame_width: self.width,
            frame_height: self.height,
        }
    }

    /// Get the current SDF scene (for rendering)
    pub fn current_scene(&self) -> Option<&SdfSceneDescriptor> {
        self.current_scene.as_ref()
    }

    /// Get the current person mask
    pub fn current_mask(&self) -> Option<&PersonMask> {
        self.current_mask.as_ref()
    }

    /// Frames received so far
    pub fn frames_received(&self) -> u32 {
        self.frames_received
    }
}

impl Default for HybridReceiver {
    fn default() -> Self {
        Self::new()
    }
}

/// Instructions for compositing the final frame on the receiver side
#[derive(Debug, Clone)]
pub struct CompositeInstruction {
    /// Whether to render the SDF background
    pub render_sdf_background: bool,
    /// SDF scene version to render
    pub sdf_scene_version: u32,
    /// Person bounding box [x, y, width, height] (None if no person)
    pub person_bbox: Option<[u32; 4]>,
    /// Person video data size in bytes
    pub person_video_size: usize,
    /// Frame dimensions
    pub frame_width: u32,
    pub frame_height: u32,
}

/// Create a person mask from segmentation data for hybrid streaming.
///
/// Convenience function that bridges ALICE-Codec segmentation with
/// ALICE-Streaming-Protocol's PersonMask format.
pub fn create_person_mask(
    binary_mask: &[u8],
    width: u32,
    height: u32,
    bbox: [u32; 4],
    foreground_count: u32,
) -> PersonMask {
    let rle = rle_encode_mask(binary_mask, width, height);
    PersonMask {
        bbox,
        rle_mask: rle,
        confidence: 1.0,
        foreground_pixels: foreground_count,
        total_pixels: width * height,
    }
}

/// Estimate bandwidth savings for a given frame configuration.
///
/// # Arguments
/// * `frame_width` - Frame width
/// * `frame_height` - Frame height
/// * `person_coverage` - Ratio of person pixels (0.0-1.0, typical: 0.15-0.30)
/// * `sdf_scene_bytes` - SDF scene size in bytes
/// * `wavelet_bpp` - Wavelet bits per pixel (typical: 0.5-2.0)
pub fn estimate_savings(
    frame_width: u32,
    frame_height: u32,
    person_coverage: f32,
    sdf_scene_bytes: usize,
    wavelet_bpp: f32,
) -> (f64, f64) {
    let total_pixels = frame_width as f64 * frame_height as f64;
    let person_pixels = total_pixels * person_coverage as f64;

    // Traditional: H.265 ≈ 0.5-2.0 bpp for full frame
    let traditional_bytes = (total_pixels * 1.0 / 8.0) as usize; // ~1 bpp average

    // Hybrid: SDF scene + person wavelet + mask overhead
    let person_bytes = (person_pixels * wavelet_bpp as f64 / 8.0) as usize;
    let mask_bytes = (person_pixels / 50.0) as usize; // ~2% overhead for RLE mask
    let hybrid_bytes = sdf_scene_bytes + person_bytes + mask_bytes;

    let ratio = traditional_bytes as f64 / hybrid_bytes.max(1) as f64;
    let savings = (1.0 - hybrid_bytes as f64 / traditional_bytes.max(1) as f64) * 100.0;

    (savings, ratio)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_scene() -> SdfSceneDescriptor {
        // Simulate ASDF binary with valid header
        let mut asdf = vec![b'A', b'S', b'D', b'F'];
        asdf.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]);
        asdf.extend_from_slice(&[0x03, 0x00, 0x00, 0x00]);
        asdf.extend_from_slice(&[0x00; 4]);
        // Body: simulated SDF nodes (~5KB)
        asdf.extend_from_slice(&vec![0xAB; 5000]);

        SdfSceneDescriptor::new(asdf)
            .with_bounds([-10.0, -10.0, -10.0], [10.0, 10.0, 10.0])
            .with_name("test_stage")
    }

    fn make_test_mask() -> PersonMask {
        PersonMask {
            bbox: [300, 100, 400, 600],
            rle_mask: vec![0; 200], // simulated compressed mask
            confidence: 0.95,
            foreground_pixels: 120_000,
            total_pixels: 1920 * 1080,
        }
    }

    #[test]
    fn test_hybrid_transmitter_keyframe() {
        let mut tx = HybridTransmitter::new();

        let scene = make_test_scene();
        let mask = make_test_mask();
        let person_video = vec![0xFFu8; 30_000]; // simulated wavelet data

        let frame = tx.create_keyframe(
            1920, 1080, 30.0,
            scene,
            Some(mask),
            person_video,
        );

        assert!(frame.is_keyframe);
        assert_eq!(frame.sequence, 1);
        assert!(frame.sdf_scene.is_some());
        assert!(frame.person_mask.is_some());
        assert_eq!(frame.person_video_data.len(), 30_000);

        // Check stats
        let stats = tx.stats();
        assert!(stats.sdf_scene_bytes > 0);
        assert!(stats.person_video_bytes > 0);
        assert!(stats.hybrid_total_bytes < stats.traditional_total_bytes);
    }

    #[test]
    fn test_hybrid_transmitter_delta() {
        let mut tx = HybridTransmitter::new();

        // First: keyframe
        let scene = make_test_scene();
        tx.create_keyframe(1920, 1080, 30.0, scene, None, vec![0; 10_000]);

        // Then: 29 delta frames (no SDF delta, just person)
        for i in 0..29 {
            let person = vec![0xABu8; 8_000];
            let frame = tx.create_delta_frame(None, None, person, i * 33);
            assert!(!frame.is_keyframe);
            assert!(frame.sdf_scene.is_none());
            assert!(frame.sdf_delta.is_none());
        }

        let stats = tx.stats();
        assert_eq!(stats.frame_count, 30);

        let (savings, ratio) = stats.savings();
        assert!(savings > 0.0, "Should have positive savings");
        assert!(ratio > 1.0, "Compression ratio should be > 1");
    }

    #[test]
    fn test_hybrid_receiver() {
        let mut rx = HybridReceiver::new();

        // Receive keyframe
        let scene = make_test_scene();
        let mask = make_test_mask();
        let keyframe = HybridFrame {
            sequence: 1,
            timestamp_ms: 0,
            width: 1920,
            height: 1080,
            sdf_scene: Some(scene),
            sdf_delta: None,
            person_mask: Some(mask),
            person_video_data: vec![0; 10_000],
            audio_data: Vec::new(),
            is_keyframe: true,
        };

        let instr = rx.process_frame(&keyframe);
        assert!(instr.render_sdf_background);
        assert!(instr.person_bbox.is_some());
        assert_eq!(instr.frame_width, 1920);

        // Scene should be cached
        assert!(rx.current_scene().is_some());
        assert!(rx.current_mask().is_some());

        // Receive delta frame
        let delta = HybridFrame {
            sequence: 2,
            timestamp_ms: 33,
            width: 1920,
            height: 1080,
            sdf_scene: None,
            sdf_delta: None,
            person_mask: None,
            person_video_data: vec![0; 8_000],
            audio_data: Vec::new(),
            is_keyframe: false,
        };

        let instr = rx.process_frame(&delta);
        assert!(instr.render_sdf_background); // still has cached scene
        assert_eq!(rx.frames_received(), 2);
    }

    #[test]
    fn test_estimate_savings() {
        // 1080p, person covers 20% of frame, SDF scene = 5KB
        let (savings, ratio) = estimate_savings(
            1920, 1080,
            0.20,   // 20% person
            5_000,  // 5KB SDF
            1.0,    // 1 bpp wavelet
        );

        assert!(savings > 70.0, "Should save >70%, got {:.1}%", savings);
        assert!(ratio > 3.0, "Should compress >3x, got {:.1}x", ratio);
    }

    #[test]
    fn test_create_person_mask() {
        let mut binary_mask = vec![0u8; 100];
        for i in 30..70 { binary_mask[i] = 1; }

        let mask = create_person_mask(&binary_mask, 10, 10, [3, 3, 4, 4], 40);
        assert_eq!(mask.foreground_pixels, 40);
        assert!(!mask.rle_mask.is_empty());
        assert!(mask.rle_mask.len() < binary_mask.len()); // RLE should be smaller
    }

    #[test]
    fn test_bandwidth_report() {
        let stats = HybridBandwidthStats {
            sdf_scene_bytes: 5_000,
            sdf_delta_bytes: 1_000,
            person_mask_bytes: 600,
            person_video_bytes: 240_000,
            audio_bytes: 0,
            hybrid_total_bytes: 246_600,
            traditional_total_bytes: 3_110_400,
            frame_width: 1920,
            frame_height: 1080,
            frame_count: 30,
        };

        let report = stats.report();
        assert!(report.contains("ALICE Hybrid"));
        assert!(report.contains("1920x1080"));
        assert!(report.contains("Savings:"));
    }
}
