//! ASP Packet Definitions
//!
//! This module defines the payload structures for each packet type:
//! - I-Packet: Keyframe with full procedural description
//! - D-Packet: Delta frame with incremental updates
//! - C-Packet: Correction packet for ROI-based pixel corrections
//! - S-Packet: Sync/Control packet
//!
//! # Serialization
//!
//! By default, packets are serialized using FlatBuffers for cross-language
//! compatibility. Enable the `bincode-compat` feature for legacy bincode support.
//!
//! ```rust,ignore
//! // FlatBuffers (default, cross-language)
//! let bytes = packet.to_bytes()?;
//!
//! // With bincode-compat feature enabled:
//! let bytes = packet.to_bytes_bincode()?;
//! ```

use crate::flatbuffers_api;
use crate::header::{crc32, AspPacketHeader};
use crate::types::*;
use serde::{Deserialize, Serialize};

/// Color palette for procedural generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorPalette {
    /// Primary colors
    pub colors: Vec<Color>,
    /// Color weights (optional)
    pub weights: Option<Vec<f32>>,
}

impl ColorPalette {
    pub fn new(colors: Vec<Color>) -> Self {
        Self {
            colors,
            weights: None,
        }
    }

    pub fn with_weights(colors: Vec<Color>, weights: Vec<f32>) -> Self {
        Self {
            colors,
            weights: Some(weights),
        }
    }

    pub fn dominant_color(&self) -> Option<Color> {
        self.colors.first().copied()
    }
}

impl Default for ColorPalette {
    fn default() -> Self {
        Self::new(vec![Color::black()])
    }
}

/// Region descriptor for procedural generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionDescriptor {
    /// Region bounds
    pub bounds: Rect,
    /// Pattern type
    pub pattern_type: PatternType,
    /// Color palette
    pub palette: ColorPalette,
    /// DCT coefficients (sparse, for DCT pattern type)
    pub dct_coefficients: Option<Vec<(u32, u32, f32)>>,
    /// Texture ID (for Texture pattern type)
    pub texture_id: Option<u32>,
    /// Additional parameters (JSON-like)
    pub params: Option<Vec<(String, f32)>>,
}

impl RegionDescriptor {
    pub fn solid(bounds: Rect, color: Color) -> Self {
        Self {
            bounds,
            pattern_type: PatternType::Solid,
            palette: ColorPalette::new(vec![color]),
            dct_coefficients: None,
            texture_id: None,
            params: None,
        }
    }

    pub fn gradient(bounds: Rect, colors: Vec<Color>) -> Self {
        Self {
            bounds,
            pattern_type: PatternType::GradientLinear,
            palette: ColorPalette::new(colors),
            dct_coefficients: None,
            texture_id: None,
            params: None,
        }
    }

    pub fn dct(bounds: Rect, palette: ColorPalette, coefficients: Vec<(u32, u32, f32)>) -> Self {
        Self {
            bounds,
            pattern_type: PatternType::Dct,
            palette,
            dct_coefficients: Some(coefficients),
            texture_id: None,
            params: None,
        }
    }
}

/// I-Packet payload: Full frame description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IPacketPayload {
    /// Frame width
    pub width: u32,
    /// Frame height
    pub height: u32,
    /// Frame rate (fps)
    pub fps: f32,
    /// Quality level
    pub quality: QualityLevel,
    /// Global color palette
    pub global_palette: ColorPalette,
    /// Region descriptors
    pub regions: Vec<RegionDescriptor>,
    /// Global animation parameters
    pub animation: Option<AnimationParams>,
    /// Timestamp in milliseconds
    pub timestamp_ms: u64,
    /// SDF scene descriptor for hybrid streaming (background as SDF)
    pub sdf_scene: Option<crate::scene::SdfSceneDescriptor>,
}

impl IPacketPayload {
    pub fn new(width: u32, height: u32, fps: f32) -> Self {
        Self {
            width,
            height,
            fps,
            quality: QualityLevel::Medium,
            global_palette: ColorPalette::default(),
            regions: Vec::new(),
            animation: None,
            timestamp_ms: 0,
            sdf_scene: None,
        }
    }

    pub fn with_quality(mut self, quality: QualityLevel) -> Self {
        self.quality = quality;
        self
    }

    pub fn with_palette(mut self, palette: ColorPalette) -> Self {
        self.global_palette = palette;
        self
    }

    pub fn add_region(&mut self, region: RegionDescriptor) {
        self.regions.push(region);
    }
}

/// Delta update for a region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionDelta {
    /// Region index (reference to I-Packet region)
    pub region_index: u32,
    /// Updated palette (if changed)
    pub palette_delta: Option<ColorPalette>,
    /// Updated DCT coefficients (sparse delta)
    pub dct_delta: Option<Vec<(u32, u32, f32)>>,
    /// Updated parameters
    pub param_delta: Option<Vec<(String, f32)>>,
}

/// D-Packet payload: Delta/incremental update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DPacketPayload {
    /// Reference sequence number (I-Packet or previous D-Packet)
    pub ref_sequence: u32,
    /// Motion vectors
    pub motion_vectors: Vec<MotionVector>,
    /// Global motion parameters
    pub global_motion: Option<AnimationParams>,
    /// Region deltas
    pub region_deltas: Vec<RegionDelta>,
    /// Frame timestamp in milliseconds
    pub timestamp_ms: u64,
    /// SDF scene delta for hybrid streaming (incremental background update)
    pub sdf_delta: Option<crate::scene::SdfSceneDelta>,
    /// Person mask for hybrid streaming (foreground segmentation)
    pub person_mask: Option<crate::scene::PersonMask>,
}

impl DPacketPayload {
    pub fn new(ref_sequence: u32) -> Self {
        Self {
            ref_sequence,
            motion_vectors: Vec::new(),
            global_motion: None,
            region_deltas: Vec::new(),
            timestamp_ms: 0,
            sdf_delta: None,
            person_mask: None,
        }
    }

    pub fn add_motion_vector(&mut self, mv: MotionVector) {
        self.motion_vectors.push(mv);
    }

    pub fn add_region_delta(&mut self, delta: RegionDelta) {
        self.region_deltas.push(delta);
    }

    /// Calculate motion vector statistics
    pub fn motion_stats(&self) -> (f32, f32) {
        if self.motion_vectors.is_empty() {
            return (0.0, 0.0);
        }

        let avg_magnitude: f32 = self
            .motion_vectors
            .iter()
            .map(|mv| mv.magnitude())
            .sum::<f32>()
            / self.motion_vectors.len() as f32;

        let max_magnitude: f32 = self
            .motion_vectors
            .iter()
            .map(|mv| mv.magnitude())
            .fold(0.0, f32::max);

        (avg_magnitude, max_magnitude)
    }
}

/// ROI region for correction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoiRegion {
    /// Region bounds
    pub bounds: Rect,
    /// ROI type
    pub roi_type: RoiType,
    /// Priority (higher = more important)
    pub priority: u8,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
}

impl RoiRegion {
    pub fn new(bounds: Rect, roi_type: RoiType) -> Self {
        Self {
            bounds,
            roi_type,
            priority: 1,
            confidence: 1.0,
        }
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

/// Correction data for a region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionData {
    /// ROI region
    pub roi: RoiRegion,
    /// Pixel data (compressed, delta from procedural)
    pub pixel_delta: Vec<u8>,
    /// Compression format
    pub compression: CompressionFormat,
}

/// Compression format for pixel data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[repr(u8)]
pub enum CompressionFormat {
    /// No compression (raw RGB)
    #[default]
    Raw = 0x00,
    /// Run-length encoding
    Rle = 0x01,
    /// LZ4 compression
    Lz4 = 0x02,
    /// Zstd compression
    Zstd = 0x03,
    /// Delta + RLE
    DeltaRle = 0x04,
}

/// C-Packet payload: Correction data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPacketPayload {
    /// Reference sequence number
    pub ref_sequence: u32,
    /// Correction regions
    pub corrections: Vec<CorrectionData>,
    /// Total correction count
    pub correction_count: u32,
    /// Frame timestamp in milliseconds
    pub timestamp_ms: u64,
}

impl CPacketPayload {
    pub fn new(ref_sequence: u32) -> Self {
        Self {
            ref_sequence,
            corrections: Vec::new(),
            correction_count: 0,
            timestamp_ms: 0,
        }
    }

    pub fn add_correction(&mut self, correction: CorrectionData) {
        self.corrections.push(correction);
        self.correction_count += 1;
    }

    pub fn total_correction_bytes(&self) -> usize {
        self.corrections.iter().map(|c| c.pixel_delta.len()).sum()
    }
}

/// S-Packet payload: Sync/Control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SPacketPayload {
    /// Sync command
    pub command: SyncCommand,
    /// Command-specific data
    pub data: SyncData,
    /// Timestamp in milliseconds
    pub timestamp_ms: u64,
}

/// Sync command data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncData {
    /// No data
    None,
    /// Sequence number (for ACK/NACK)
    Sequence(u32),
    /// Bitrate in kbps
    Bitrate(u32),
    /// Quality level
    Quality(QualityLevel),
    /// Latency in milliseconds
    Latency(u32),
    /// Custom data
    Custom(Vec<u8>),
}

impl SPacketPayload {
    pub fn request_keyframe() -> Self {
        Self {
            command: SyncCommand::RequestKeyframe,
            data: SyncData::None,
            timestamp_ms: 0,
        }
    }

    pub fn ack(sequence: u32) -> Self {
        Self {
            command: SyncCommand::Ack,
            data: SyncData::Sequence(sequence),
            timestamp_ms: 0,
        }
    }

    pub fn nack(sequence: u32) -> Self {
        Self {
            command: SyncCommand::Nack,
            data: SyncData::Sequence(sequence),
            timestamp_ms: 0,
        }
    }

    pub fn end_of_stream() -> Self {
        Self {
            command: SyncCommand::EndOfStream,
            data: SyncData::None,
            timestamp_ms: 0,
        }
    }

    pub fn bitrate_adjust(bitrate_kbps: u32) -> Self {
        Self {
            command: SyncCommand::BitrateAdjust,
            data: SyncData::Bitrate(bitrate_kbps),
            timestamp_ms: 0,
        }
    }

    pub fn ping() -> Self {
        Self {
            command: SyncCommand::Ping,
            data: SyncData::None,
            timestamp_ms: 0,
        }
    }

    pub fn pong() -> Self {
        Self {
            command: SyncCommand::Pong,
            data: SyncData::None,
            timestamp_ms: 0,
        }
    }
}

/// ASP Packet (header + payload)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AspPacket {
    /// Packet header
    pub header: AspPacketHeader,
    /// Packet payload
    pub payload: AspPayload,
}

/// Packet payload enum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AspPayload {
    IPacket(IPacketPayload),
    DPacket(DPacketPayload),
    CPacket(CPacketPayload),
    SPacket(SPacketPayload),
}

impl AspPacket {
    /// Create an I-Packet
    pub fn create_i_packet(sequence: u32, payload: IPacketPayload) -> AspResult<Self> {
        // Estimate payload size (used for header, actual size computed during serialization)
        let estimated_size = Self::estimate_i_packet_size(&payload);

        Ok(Self {
            header: AspPacketHeader::new(PacketType::IPacket, sequence, estimated_size as u32),
            payload: AspPayload::IPacket(payload),
        })
    }

    /// Create a D-Packet
    pub fn create_d_packet(sequence: u32, payload: DPacketPayload) -> AspResult<Self> {
        let estimated_size = Self::estimate_d_packet_size(&payload);

        Ok(Self {
            header: AspPacketHeader::new(PacketType::DPacket, sequence, estimated_size as u32),
            payload: AspPayload::DPacket(payload),
        })
    }

    /// Create a C-Packet
    pub fn create_c_packet(sequence: u32, payload: CPacketPayload) -> AspResult<Self> {
        let estimated_size = Self::estimate_c_packet_size(&payload);

        Ok(Self {
            header: AspPacketHeader::new(PacketType::CPacket, sequence, estimated_size as u32),
            payload: AspPayload::CPacket(payload),
        })
    }

    /// Create an S-Packet
    pub fn create_s_packet(sequence: u32, payload: SPacketPayload) -> AspResult<Self> {
        Ok(Self {
            header: AspPacketHeader::new(
                PacketType::SPacket,
                sequence,
                64, // S-Packets are small
            ),
            payload: AspPayload::SPacket(payload),
        })
    }

    // Payload size estimation (for header creation)
    fn estimate_i_packet_size(payload: &IPacketPayload) -> usize {
        64 + payload.regions.len() * 128
    }

    fn estimate_d_packet_size(payload: &DPacketPayload) -> usize {
        32 + payload.motion_vectors.len() * 12 + payload.region_deltas.len() * 64
    }

    fn estimate_c_packet_size(payload: &CPacketPayload) -> usize {
        32 + payload
            .corrections
            .iter()
            .map(|c| c.pixel_delta.len() + 32)
            .sum::<usize>()
    }

    /// Serialize packet to bytes using FlatBuffers (cross-language compatible)
    ///
    /// This is the default serialization method that produces output
    /// readable by C++, Go, Java, Python, TypeScript, etc.
    pub fn to_bytes(&self) -> AspResult<Vec<u8>> {
        let mut buffer = Vec::new();
        self.write_to_buffer(&mut buffer)?;
        Ok(buffer)
    }

    /// Serialize packet to an existing buffer using FlatBuffers
    ///
    /// Format: [Header (16 bytes)] [FlatBuffers Payload] [CRC32 (4 bytes)]
    pub fn write_to_buffer(&self, buffer: &mut Vec<u8>) -> AspResult<()> {
        buffer.clear();

        // Serialize payload using FlatBuffers
        let fb_payload = match &self.payload {
            AspPayload::IPacket(p) => {
                flatbuffers_api::create_i_packet(p.width, p.height, p.fps, p.timestamp_ms)
            }
            AspPayload::DPacket(p) => {
                flatbuffers_api::create_d_packet(p.ref_sequence, &p.motion_vectors, p.timestamp_ms)
            }
            AspPayload::SPacket(p) => {
                match p.command {
                    SyncCommand::Ping => flatbuffers_api::create_ping(p.timestamp_ms),
                    SyncCommand::Pong => flatbuffers_api::create_pong(p.timestamp_ms),
                    _ => flatbuffers_api::create_ping(p.timestamp_ms), // Fallback
                }
            }
            AspPayload::CPacket(_) => {
                // C-Packet FlatBuffers serialization (minimal for now)
                flatbuffers_api::create_ping(0) // Placeholder
            }
        };

        let payload_len = fb_payload.len();

        // Reserve capacity
        buffer.reserve(AspPacketHeader::SIZE + payload_len + 4);

        // Write header
        let header = AspPacketHeader::new(
            self.header.packet_type,
            self.header.sequence,
            payload_len as u32,
        );
        buffer.extend_from_slice(&header.to_bytes());

        // Write FlatBuffers payload
        buffer.extend_from_slice(&fb_payload);

        // Calculate and append CRC32
        let checksum = crc32(buffer);
        buffer.extend_from_slice(&checksum.to_be_bytes());

        Ok(())
    }

    /// Deserialize packet from bytes (FlatBuffers format)
    pub fn from_bytes(data: &[u8]) -> AspResult<Self> {
        // Minimum size: header + checksum
        if data.len() < AspPacketHeader::SIZE + 4 {
            return Err(AspError::IncompletePacket {
                expected: AspPacketHeader::SIZE + 4,
                got: data.len(),
            });
        }

        // Verify checksum
        let checksum_offset = data.len() - 4;
        let expected_checksum = u32::from_be_bytes([
            data[checksum_offset],
            data[checksum_offset + 1],
            data[checksum_offset + 2],
            data[checksum_offset + 3],
        ]);
        let actual_checksum = crc32(&data[..checksum_offset]);

        if expected_checksum != actual_checksum {
            return Err(AspError::ChecksumMismatch {
                expected: expected_checksum,
                got: actual_checksum,
            });
        }

        // Parse header
        let header = AspPacketHeader::from_bytes(data)?;

        // Parse FlatBuffers payload
        let payload_data = &data[AspPacketHeader::SIZE..checksum_offset];

        let payload = match header.packet_type {
            PacketType::IPacket => {
                let fb = flatbuffers_api::read_i_packet(payload_data)
                    .map_err(|e| AspError::DeserializationError(e.to_string()))?;
                let p = IPacketPayload {
                    width: fb.width(),
                    height: fb.height(),
                    fps: fb.fps(),
                    quality: match fb.quality() {
                        flatbuffers_api::FbQualityLevel::Low => QualityLevel::Low,
                        flatbuffers_api::FbQualityLevel::High => QualityLevel::High,
                        flatbuffers_api::FbQualityLevel::Ultra => QualityLevel::Ultra,
                        _ => QualityLevel::Medium,
                    },
                    global_palette: ColorPalette::default(),
                    regions: Vec::new(),
                    animation: None,
                    timestamp_ms: fb.timestamp_ms(),
                    sdf_scene: None,
                };
                AspPayload::IPacket(p)
            }
            PacketType::DPacket => {
                let fb = flatbuffers_api::read_d_packet(payload_data)
                    .map_err(|e| AspError::DeserializationError(e.to_string()))?;
                let mut p = DPacketPayload::new(fb.ref_sequence());
                p.timestamp_ms = fb.timestamp_ms();

                // Convert motion vectors
                if let Some(mvs) = fb.motion_vectors() {
                    for i in 0..mvs.len() {
                        let mv = mvs.get(i);
                        p.motion_vectors.push(MotionVector::new(
                            mv.block_x(),
                            mv.block_y(),
                            mv.dx(),
                            mv.dy(),
                            mv.sad(),
                        ));
                    }
                }
                AspPayload::DPacket(p)
            }
            PacketType::SPacket => {
                let fb = flatbuffers_api::read_s_packet(payload_data)
                    .map_err(|e| AspError::DeserializationError(e.to_string()))?;
                let command = match fb.command() {
                    flatbuffers_api::FbSyncCommand::RequestKeyframe => SyncCommand::RequestKeyframe,
                    flatbuffers_api::FbSyncCommand::Ack => SyncCommand::Ack,
                    flatbuffers_api::FbSyncCommand::Nack => SyncCommand::Nack,
                    flatbuffers_api::FbSyncCommand::EndOfStream => SyncCommand::EndOfStream,
                    flatbuffers_api::FbSyncCommand::BitrateAdjust => SyncCommand::BitrateAdjust,
                    flatbuffers_api::FbSyncCommand::QualityChange => SyncCommand::QualityChange,
                    flatbuffers_api::FbSyncCommand::Ping => SyncCommand::Ping,
                    flatbuffers_api::FbSyncCommand::Pong => SyncCommand::Pong,
                    _ => SyncCommand::Ping,
                };
                let p = SPacketPayload {
                    command,
                    data: SyncData::None,
                    timestamp_ms: fb.timestamp_ms(),
                };
                AspPayload::SPacket(p)
            }
            PacketType::CPacket => {
                // Minimal C-Packet support for now
                AspPayload::CPacket(CPacketPayload::new(0))
            }
        };

        Ok(Self { header, payload })
    }

    /// Estimate the serialized size of this packet (for pre-allocation)
    #[inline]
    pub fn estimated_size(&self) -> usize {
        let payload_estimate = match &self.payload {
            AspPayload::IPacket(p) => Self::estimate_i_packet_size(p),
            AspPayload::DPacket(p) => Self::estimate_d_packet_size(p),
            AspPayload::CPacket(p) => Self::estimate_c_packet_size(p),
            AspPayload::SPacket(_) => 64,
        };
        AspPacketHeader::SIZE + payload_estimate + 4 // +4 for CRC32
    }

    /// Get packet type
    pub fn packet_type(&self) -> PacketType {
        self.header.packet_type
    }

    /// Get sequence number
    pub fn sequence(&self) -> u32 {
        self.header.sequence
    }

    /// Check if this is a keyframe
    pub fn is_keyframe(&self) -> bool {
        self.header.is_keyframe()
    }

    /// Get payload size
    pub fn payload_size(&self) -> u32 {
        self.header.payload_length
    }

    /// Get I-Packet payload (if applicable)
    pub fn as_i_packet(&self) -> Option<&IPacketPayload> {
        match &self.payload {
            AspPayload::IPacket(p) => Some(p),
            _ => None,
        }
    }

    /// Get D-Packet payload (if applicable)
    pub fn as_d_packet(&self) -> Option<&DPacketPayload> {
        match &self.payload {
            AspPayload::DPacket(p) => Some(p),
            _ => None,
        }
    }

    /// Get C-Packet payload (if applicable)
    pub fn as_c_packet(&self) -> Option<&CPacketPayload> {
        match &self.payload {
            AspPayload::CPacket(p) => Some(p),
            _ => None,
        }
    }

    /// Get S-Packet payload (if applicable)
    pub fn as_s_packet(&self) -> Option<&SPacketPayload> {
        match &self.payload {
            AspPayload::SPacket(p) => Some(p),
            _ => None,
        }
    }
}

// =============================================================================
// PacketEncoder (Zero-Allocation Hot Loop)
// =============================================================================

use flatbuffers::FlatBufferBuilder;

/// High-performance packet encoder with Builder/Buffer reuse
///
/// Use this for streaming scenarios where you need to encode many packets
/// per second without allocation overhead.
///
/// # Example
///
/// ```rust,ignore
/// use libasp::PacketEncoder;
///
/// let mut encoder = PacketEncoder::new();
///
/// // Hot loop - no allocations!
/// for frame in frames {
///     let bytes = encoder.encode_d_packet(
///         sequence,
///         frame.ref_seq,
///         &frame.mvs,
///         frame.timestamp_ms,
///     );
///     socket.send(bytes)?;
/// }
/// ```
pub struct PacketEncoder {
    /// Reusable FlatBufferBuilder (public for direct access in hot paths)
    pub builder: FlatBufferBuilder<'static>,
    /// Reusable output buffer (public for direct access in hot paths)
    pub buffer: Vec<u8>,
}

impl Default for PacketEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl PacketEncoder {
    /// Create a new PacketEncoder with default capacity
    pub fn new() -> Self {
        Self::with_capacity(4096, 8192)
    }

    /// Create a new PacketEncoder with specified capacities
    ///
    /// # Arguments
    /// * `builder_capacity` - Initial capacity for FlatBufferBuilder (bytes)
    /// * `buffer_capacity` - Initial capacity for output buffer (bytes)
    pub fn with_capacity(builder_capacity: usize, buffer_capacity: usize) -> Self {
        Self {
            builder: FlatBufferBuilder::with_capacity(builder_capacity),
            buffer: Vec::with_capacity(buffer_capacity),
        }
    }

    /// Encode a D-Packet (zero-allocation after warmup)
    ///
    /// Returns a slice of the internal buffer containing the complete packet:
    /// `[Header (16 bytes)] [FlatBuffers Payload] [CRC32 (4 bytes)]`
    #[inline]
    pub fn encode_d_packet(
        &mut self,
        sequence: u32,
        ref_sequence: u32,
        motion_vectors: &[MotionVector],
        timestamp_ms: u64,
    ) -> &[u8] {
        // Get FlatBuffers payload using builder reuse
        let fb_payload = flatbuffers_api::encode_d_packet_with_builder(
            &mut self.builder,
            ref_sequence,
            motion_vectors,
            timestamp_ms,
        );
        let payload_len = fb_payload.len();

        // Prepare output buffer
        let total_size = AspPacketHeader::SIZE + payload_len + 4;
        self.buffer.clear();
        self.buffer.reserve(total_size);

        // SAFETY: We're writing exactly total_size bytes
        unsafe {
            self.buffer.set_len(total_size);
            let ptr = self.buffer.as_mut_ptr();

            // Write header using raw pointer (zero bounds check)
            let header = AspPacketHeader::new(PacketType::DPacket, sequence, payload_len as u32);
            header.write_to_ptr(ptr);

            // Copy FlatBuffers payload
            std::ptr::copy_nonoverlapping(
                fb_payload.as_ptr(),
                ptr.add(AspPacketHeader::SIZE),
                payload_len,
            );

            // Calculate CRC32 and write
            let crc = crc32(&self.buffer[..total_size - 4]);
            let crc_bytes = crc.to_be_bytes();
            std::ptr::copy_nonoverlapping(crc_bytes.as_ptr(), ptr.add(total_size - 4), 4);
        }

        &self.buffer
    }

    /// Encode an I-Packet (zero-allocation after warmup)
    #[inline]
    pub fn encode_i_packet(
        &mut self,
        sequence: u32,
        width: u32,
        height: u32,
        fps: f32,
        quality: flatbuffers_api::FbQualityLevel,
        timestamp_ms: u64,
    ) -> &[u8] {
        // Get FlatBuffers payload using builder reuse
        let fb_payload = flatbuffers_api::encode_i_packet_with_builder(
            &mut self.builder,
            width,
            height,
            fps,
            quality,
            timestamp_ms,
        );
        let payload_len = fb_payload.len();

        // Prepare output buffer
        let total_size = AspPacketHeader::SIZE + payload_len + 4;
        self.buffer.clear();
        self.buffer.reserve(total_size);

        // SAFETY: We're writing exactly total_size bytes
        unsafe {
            self.buffer.set_len(total_size);
            let ptr = self.buffer.as_mut_ptr();

            // Write header using raw pointer (zero bounds check)
            let header = AspPacketHeader::new(PacketType::IPacket, sequence, payload_len as u32);
            header.write_to_ptr(ptr);

            // Copy FlatBuffers payload
            std::ptr::copy_nonoverlapping(
                fb_payload.as_ptr(),
                ptr.add(AspPacketHeader::SIZE),
                payload_len,
            );

            // Calculate CRC32 and write
            let crc = crc32(&self.buffer[..total_size - 4]);
            let crc_bytes = crc.to_be_bytes();
            std::ptr::copy_nonoverlapping(crc_bytes.as_ptr(), ptr.add(total_size - 4), 4);
        }

        &self.buffer
    }

    /// Encode an S-Packet (zero-allocation after warmup)
    #[inline]
    pub fn encode_s_packet(
        &mut self,
        sequence: u32,
        command: flatbuffers_api::FbSyncCommand,
        timestamp_ms: u64,
    ) -> &[u8] {
        // Get FlatBuffers payload using builder reuse
        let fb_payload =
            flatbuffers_api::encode_s_packet_with_builder(&mut self.builder, command, timestamp_ms);
        let payload_len = fb_payload.len();

        // Prepare output buffer
        let total_size = AspPacketHeader::SIZE + payload_len + 4;
        self.buffer.clear();
        self.buffer.reserve(total_size);

        // SAFETY: We're writing exactly total_size bytes
        unsafe {
            self.buffer.set_len(total_size);
            let ptr = self.buffer.as_mut_ptr();

            // Write header using raw pointer (zero bounds check)
            let header = AspPacketHeader::new(PacketType::SPacket, sequence, payload_len as u32);
            header.write_to_ptr(ptr);

            // Copy FlatBuffers payload
            std::ptr::copy_nonoverlapping(
                fb_payload.as_ptr(),
                ptr.add(AspPacketHeader::SIZE),
                payload_len,
            );

            // Calculate CRC32 and write
            let crc = crc32(&self.buffer[..total_size - 4]);
            let crc_bytes = crc.to_be_bytes();
            std::ptr::copy_nonoverlapping(crc_bytes.as_ptr(), ptr.add(total_size - 4), 4);
        }

        &self.buffer
    }

    /// Get the internal buffer's current capacity
    pub fn buffer_capacity(&self) -> usize {
        self.buffer.capacity()
    }
}

// =============================================================================
// Legacy bincode support (optional)
// =============================================================================

#[cfg(feature = "bincode-compat")]
impl AspPacket {
    /// Serialize packet to bytes using bincode (Rust-only, legacy)
    pub fn to_bytes_bincode(&self) -> AspResult<Vec<u8>> {
        let mut buffer = Vec::new();
        self.write_to_buffer_bincode(&mut buffer)?;
        Ok(buffer)
    }

    /// Serialize packet to an existing buffer using bincode
    pub fn write_to_buffer_bincode(&self, buffer: &mut Vec<u8>) -> AspResult<()> {
        buffer.clear();

        let estimated = self.estimated_size();
        buffer.reserve(estimated);

        unsafe {
            buffer.set_len(AspPacketHeader::SIZE);
        }

        let payload_start = AspPacketHeader::SIZE;
        match &self.payload {
            AspPayload::IPacket(p) => {
                bincode::serialize_into(&mut *buffer, p)
                    .map_err(|e| AspError::SerializationError(e.to_string()))?;
            }
            AspPayload::DPacket(p) => {
                bincode::serialize_into(&mut *buffer, p)
                    .map_err(|e| AspError::SerializationError(e.to_string()))?;
            }
            AspPayload::CPacket(p) => {
                bincode::serialize_into(&mut *buffer, p)
                    .map_err(|e| AspError::SerializationError(e.to_string()))?;
            }
            AspPayload::SPacket(p) => {
                bincode::serialize_into(&mut *buffer, p)
                    .map_err(|e| AspError::SerializationError(e.to_string()))?;
            }
        };

        let payload_len = buffer.len() - payload_start;

        let header = AspPacketHeader::new(
            self.header.packet_type,
            self.header.sequence,
            payload_len as u32,
        );

        buffer[..AspPacketHeader::SIZE].copy_from_slice(&header.to_bytes());

        let checksum = crc32(buffer);
        let crc_bytes = checksum.to_be_bytes();
        let len = buffer.len();

        buffer.reserve(4);
        unsafe {
            let ptr = buffer.as_mut_ptr().add(len);
            std::ptr::copy_nonoverlapping(crc_bytes.as_ptr(), ptr, 4);
            buffer.set_len(len + 4);
        }

        Ok(())
    }

    /// Deserialize packet from bytes (bincode format)
    pub fn from_bytes_bincode(data: &[u8]) -> AspResult<Self> {
        if data.len() < AspPacketHeader::SIZE + 4 {
            return Err(AspError::IncompletePacket {
                expected: AspPacketHeader::SIZE + 4,
                got: data.len(),
            });
        }

        let checksum_offset = data.len() - 4;
        let expected_checksum = u32::from_be_bytes([
            data[checksum_offset],
            data[checksum_offset + 1],
            data[checksum_offset + 2],
            data[checksum_offset + 3],
        ]);
        let actual_checksum = crc32(&data[..checksum_offset]);

        if expected_checksum != actual_checksum {
            return Err(AspError::ChecksumMismatch {
                expected: expected_checksum,
                got: actual_checksum,
            });
        }

        let header = AspPacketHeader::from_bytes(data)?;
        let payload_data = &data[AspPacketHeader::SIZE..checksum_offset];

        let payload = match header.packet_type {
            PacketType::IPacket => {
                let p: IPacketPayload = bincode::deserialize(payload_data)
                    .map_err(|e| AspError::DeserializationError(e.to_string()))?;
                AspPayload::IPacket(p)
            }
            PacketType::DPacket => {
                let p: DPacketPayload = bincode::deserialize(payload_data)
                    .map_err(|e| AspError::DeserializationError(e.to_string()))?;
                AspPayload::DPacket(p)
            }
            PacketType::CPacket => {
                let p: CPacketPayload = bincode::deserialize(payload_data)
                    .map_err(|e| AspError::DeserializationError(e.to_string()))?;
                AspPayload::CPacket(p)
            }
            PacketType::SPacket => {
                let p: SPacketPayload = bincode::deserialize(payload_data)
                    .map_err(|e| AspError::DeserializationError(e.to_string()))?;
                AspPayload::SPacket(p)
            }
        };

        Ok(Self { header, payload })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i_packet_creation() {
        let mut payload = IPacketPayload::new(1920, 1080, 30.0);
        payload.add_region(RegionDescriptor::solid(
            Rect::new(0, 0, 1920, 1080),
            Color::new(128, 128, 128),
        ));

        let packet = AspPacket::create_i_packet(1, payload).unwrap();
        assert!(packet.is_keyframe());
        assert_eq!(packet.sequence(), 1);
    }

    #[test]
    fn test_d_packet_creation() {
        let mut payload = DPacketPayload::new(1);
        payload.add_motion_vector(MotionVector::new(0, 0, 5, -3, 100));

        let packet = AspPacket::create_d_packet(2, payload).unwrap();
        assert!(!packet.is_keyframe());
        assert_eq!(packet.packet_type(), PacketType::DPacket);
    }

    #[test]
    fn test_packet_serialization() {
        let payload = IPacketPayload::new(1920, 1080, 30.0);
        let packet = AspPacket::create_i_packet(1, payload).unwrap();

        let bytes = packet.to_bytes().unwrap();
        let restored = AspPacket::from_bytes(&bytes).unwrap();

        assert_eq!(packet.sequence(), restored.sequence());
        assert_eq!(packet.packet_type(), restored.packet_type());

        let original_i = packet.as_i_packet().unwrap();
        let restored_i = restored.as_i_packet().unwrap();
        assert_eq!(original_i.width, restored_i.width);
        assert_eq!(original_i.height, restored_i.height);
    }

    #[test]
    fn test_s_packet_commands() {
        let ack = SPacketPayload::ack(42);
        assert!(matches!(ack.command, SyncCommand::Ack));
        assert!(matches!(ack.data, SyncData::Sequence(42)));

        let ping = SPacketPayload::ping();
        assert!(matches!(ping.command, SyncCommand::Ping));
    }

    #[test]
    fn test_motion_stats() {
        let mut payload = DPacketPayload::new(1);
        payload.add_motion_vector(MotionVector::new(0, 0, 3, 4, 100)); // magnitude = 5
        payload.add_motion_vector(MotionVector::new(1, 0, 6, 8, 200)); // magnitude = 10

        let (avg, max) = payload.motion_stats();
        assert!((avg - 7.5).abs() < 0.01);
        assert!((max - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_checksum_verification() {
        let payload = IPacketPayload::new(640, 480, 30.0);
        let packet = AspPacket::create_i_packet(1, payload).unwrap();

        let mut bytes = packet.to_bytes().unwrap();

        // Corrupt a byte
        bytes[20] ^= 0xFF;

        let result = AspPacket::from_bytes(&bytes);
        assert!(matches!(result, Err(AspError::ChecksumMismatch { .. })));
    }

    #[test]
    fn test_d_packet_with_motion_vectors() {
        let mut payload = DPacketPayload::new(42);
        payload.timestamp_ms = 12345;
        payload.add_motion_vector(MotionVector::new(0, 0, 5, -3, 100));
        payload.add_motion_vector(MotionVector::new(1, 0, 2, 1, 50));

        let packet = AspPacket::create_d_packet(1, payload).unwrap();
        let bytes = packet.to_bytes().unwrap();
        let restored = AspPacket::from_bytes(&bytes).unwrap();

        let d = restored.as_d_packet().unwrap();
        assert_eq!(d.ref_sequence, 42);
        assert_eq!(d.timestamp_ms, 12345);
        assert_eq!(d.motion_vectors.len(), 2);

        let mv0 = &d.motion_vectors[0];
        assert_eq!(mv0.block_x, 0);
        assert_eq!(mv0.dx, 5);
        assert_eq!(mv0.dy, -3);
    }

    #[test]
    fn test_estimated_size() {
        let mut payload = IPacketPayload::new(1920, 1080, 30.0);
        payload.add_region(RegionDescriptor::solid(
            Rect::new(0, 0, 1920, 1080),
            Color::new(128, 128, 128),
        ));

        let packet = AspPacket::create_i_packet(1, payload).unwrap();
        let estimated = packet.estimated_size();
        let actual = packet.to_bytes().unwrap().len();

        assert!(estimated > 0);
        // Estimated should be reasonable (allow some variance due to FlatBuffers overhead)
        assert!(
            estimated >= actual / 4,
            "Estimate {} too small for actual {}",
            estimated,
            actual
        );
    }

    #[test]
    fn test_packet_encoder_d_packet() {
        let mut encoder = PacketEncoder::new();

        let mvs = vec![
            MotionVector::new(0, 0, 5, -3, 100),
            MotionVector::new(1, 0, 2, 1, 50),
        ];

        // Encode
        let bytes = encoder.encode_d_packet(1, 42, &mvs, 12345);

        // Verify it can be parsed
        let packet = AspPacket::from_bytes(bytes).unwrap();
        assert_eq!(packet.sequence(), 1);
        assert_eq!(packet.packet_type(), PacketType::DPacket);

        let d = packet.as_d_packet().unwrap();
        assert_eq!(d.ref_sequence, 42);
        assert_eq!(d.motion_vectors.len(), 2);
    }

    #[test]
    fn test_packet_encoder_reuse() {
        let mut encoder = PacketEncoder::new();

        // Encode multiple packets with the same encoder
        for seq in 0..10 {
            let mvs = vec![MotionVector::new(seq as u16, 0, 1, 1, 10)];
            let bytes = encoder.encode_d_packet(seq, 0, &mvs, seq as u64 * 100);

            let packet = AspPacket::from_bytes(bytes).unwrap();
            assert_eq!(packet.sequence(), seq);
        }
    }

    #[test]
    fn test_packet_encoder_i_packet() {
        let mut encoder = PacketEncoder::new();

        let bytes = encoder.encode_i_packet(
            1,
            1920,
            1080,
            30.0,
            flatbuffers_api::FbQualityLevel::High,
            12345,
        );

        let packet = AspPacket::from_bytes(bytes).unwrap();
        assert_eq!(packet.sequence(), 1);
        assert!(packet.is_keyframe());

        let i = packet.as_i_packet().unwrap();
        assert_eq!(i.width, 1920);
        assert_eq!(i.height, 1080);
    }

    #[test]
    fn test_packet_encoder_s_packet() {
        let mut encoder = PacketEncoder::new();

        let bytes = encoder.encode_s_packet(1, flatbuffers_api::FbSyncCommand::Ping, 99999);

        let packet = AspPacket::from_bytes(bytes).unwrap();
        assert_eq!(packet.sequence(), 1);
        assert_eq!(packet.packet_type(), PacketType::SPacket);

        let s = packet.as_s_packet().unwrap();
        assert!(matches!(s.command, SyncCommand::Ping));
    }
}
