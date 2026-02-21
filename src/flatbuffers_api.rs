//! FlatBuffers API for ASP Protocol
//!
//! This module provides a high-level, ergonomic API for FlatBuffers serialization.
//! It wraps the auto-generated FlatBuffers code with idiomatic Rust interfaces.
//!
//! # Zero-Copy Design
//!
//! FlatBuffers enables true zero-copy access to serialized data. Unlike traditional
//! serialization formats (JSON, Protobuf), FlatBuffers allows direct access to
//! fields without parsing or copying.
//!
//! # Cross-Language Compatibility
//!
//! Data serialized with this module can be read by any language with FlatBuffers
//! support (C++, Go, Java, Python, TypeScript, etc.) using the same schema.
//!
//! # Example
//!
//! ```rust,ignore
//! use libasp::flatbuffers_api::{DPacketBuilder, read_d_packet};
//! use libasp::MotionVector;
//!
//! // Build a D-Packet
//! let mvs = vec![
//!     MotionVector::new(0, 0, 5, -3, 100),
//!     MotionVector::new(1, 0, 2, 1, 50),
//! ];
//! let bytes = DPacketBuilder::new(1)
//!     .motion_vectors(&mvs)
//!     .timestamp_ms(12345)
//!     .build();
//!
//! // Read the packet (zero-copy access!)
//! let packet = read_d_packet(&bytes).unwrap();
//! let ref_seq = packet.ref_sequence();
//! let motion_vectors = packet.motion_vectors().unwrap();
//! for mv in motion_vectors.iter() {
//!     println!("Block ({}, {}): dx={}, dy={}", mv.block_x(), mv.block_y(), mv.dx(), mv.dy());
//! }
//! ```

use crate::generated;
use crate::types::{Color as RustColor, MotionVector as RustMotionVector, Rect as RustRect};
use flatbuffers::FlatBufferBuilder;

// Re-export FlatBuffers types for convenience
pub use generated::{
    AspPacketPayload,
    CPacketPayload,
    // Structs (zero-copy accessible)
    Color as FbColor,
    CompressionFormat as FbCompressionFormat,
    DPacketPayload,
    DctCoefficient as FbDctCoefficient,

    EasingType as FbEasingType,

    // Tables (need accessor methods)
    IPacketPayload,
    MotionType as FbMotionType,
    MotionVector as FbMotionVector,
    MotionVectorCompact as FbMotionVectorCompact,
    PacketType as FbPacketType,
    PatternType as FbPatternType,
    Point as FbPoint,
    QualityLevel as FbQualityLevel,
    Rect as FbRect,
    RoiType as FbRoiType,
    SPacketPayload,
    SyncCommand as FbSyncCommand,
};

/// Error type for FlatBuffers operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum FlatBuffersError {
    /// Invalid buffer format
    #[error("Invalid FlatBuffers format: {0}")]
    InvalidFormat(String),

    /// Missing required field
    #[error("Missing required field: {0}")]
    MissingField(&'static str),

    /// Buffer too small
    #[error("Buffer too small: need {needed}, got {got}")]
    BufferTooSmall { needed: usize, got: usize },
}

/// Result type for FlatBuffers operations
pub type FbResult<T> = Result<T, FlatBuffersError>;

// =============================================================================
// Type Conversions
// =============================================================================

/// Convert Rust MotionVector to FlatBuffers MotionVector
#[inline]
pub fn motion_vector_to_fb(mv: &RustMotionVector) -> FbMotionVector {
    FbMotionVector::new(mv.block_x, mv.block_y, mv.dx, mv.dy, mv.sad)
}

/// Convert FlatBuffers MotionVector to Rust MotionVector
#[inline]
pub fn motion_vector_from_fb(mv: &FbMotionVector) -> RustMotionVector {
    RustMotionVector::new(mv.block_x(), mv.block_y(), mv.dx(), mv.dy(), mv.sad())
}

/// Convert Rust Color to FlatBuffers Color
#[inline]
pub fn color_to_fb(c: &RustColor) -> FbColor {
    FbColor::new(c.r, c.g, c.b)
}

/// Convert FlatBuffers Color to Rust Color
#[inline]
pub fn color_from_fb(c: &FbColor) -> RustColor {
    RustColor::new(c.r(), c.g(), c.b())
}

/// Convert Rust Rect to FlatBuffers Rect
#[inline]
pub fn rect_to_fb(r: &RustRect) -> FbRect {
    FbRect::new(r.x, r.y, r.width, r.height)
}

/// Convert FlatBuffers Rect to Rust Rect
#[inline]
pub fn rect_from_fb(r: &FbRect) -> RustRect {
    RustRect::new(r.x(), r.y(), r.width(), r.height())
}

// =============================================================================
// D-Packet Builder (Most common use case for motion vectors)
// =============================================================================

/// Builder for D-Packet (Delta packet with motion vectors)
///
/// This is the most performance-critical packet type for streaming.
/// Motion vectors are stored as structs for zero-copy access.
///
/// # Example
///
/// ```rust,ignore
/// let bytes = DPacketBuilder::new(1)
///     .motion_vectors(&mvs)
///     .timestamp_ms(12345)
///     .build();
/// ```
pub struct DPacketBuilder<'a> {
    builder: FlatBufferBuilder<'a>,
    ref_sequence: u32,
    motion_vectors: Option<Vec<FbMotionVector>>,
    compact_vectors: Option<Vec<FbMotionVectorCompact>>,
    timestamp_ms: u64,
}

impl<'a> DPacketBuilder<'a> {
    /// Create a new D-Packet builder
    pub fn new(ref_sequence: u32) -> Self {
        Self {
            builder: FlatBufferBuilder::with_capacity(1024),
            ref_sequence,
            motion_vectors: None,
            compact_vectors: None,
            timestamp_ms: 0,
        }
    }

    /// Create with pre-allocated capacity
    pub fn with_capacity(ref_sequence: u32, capacity: usize) -> Self {
        Self {
            builder: FlatBufferBuilder::with_capacity(capacity),
            ref_sequence,
            motion_vectors: None,
            compact_vectors: None,
            timestamp_ms: 0,
        }
    }

    /// Add motion vectors (full format, 12 bytes each)
    pub fn motion_vectors(mut self, mvs: &[RustMotionVector]) -> Self {
        self.motion_vectors = Some(mvs.iter().map(motion_vector_to_fb).collect());
        self
    }

    /// Add motion vectors (compact format, 2 bytes each)
    ///
    /// Use this for bandwidth-critical scenarios where block position
    /// is implicit from array index.
    pub fn compact_vectors(mut self, mvs: &[(i8, i8)]) -> Self {
        self.compact_vectors = Some(
            mvs.iter()
                .map(|(dx, dy)| FbMotionVectorCompact::new(*dx, *dy))
                .collect(),
        );
        self
    }

    /// Set timestamp in milliseconds
    pub fn timestamp_ms(mut self, ts: u64) -> Self {
        self.timestamp_ms = ts;
        self
    }

    /// Build the packet and return serialized bytes
    pub fn build(mut self) -> Vec<u8> {
        // Create motion vectors vector
        let mvs_offset = self
            .motion_vectors
            .as_ref()
            .map(|mvs| self.builder.create_vector(mvs));

        let compact_offset = self
            .compact_vectors
            .as_ref()
            .map(|mvs| self.builder.create_vector(mvs));

        // Build DPacketPayload
        let d_packet = generated::DPacketPayload::create(
            &mut self.builder,
            &generated::DPacketPayloadArgs {
                ref_sequence: self.ref_sequence,
                motion_vectors: mvs_offset,
                motion_vectors_compact: compact_offset,
                global_motion: None,
                region_deltas: None,
                timestamp_ms: self.timestamp_ms,
            },
        );

        // Wrap in AspPacketPayload union
        let asp_packet = generated::AspPacketPayload::create(
            &mut self.builder,
            &generated::AspPacketPayloadArgs {
                payload_type: generated::AspPayloadUnion::DPacketPayload,
                payload: Some(d_packet.as_union_value()),
            },
        );

        self.builder.finish(asp_packet, Some("ASP1"));
        self.builder.finished_data().to_vec()
    }

    /// Build directly into an existing buffer (zero-allocation)
    ///
    /// Returns the number of bytes written, or an error if the buffer is too small.
    pub fn build_into(mut self, buffer: &mut [u8]) -> FbResult<usize> {
        // Create motion vectors vector
        let mvs_offset = self
            .motion_vectors
            .as_ref()
            .map(|mvs| self.builder.create_vector(mvs));

        let compact_offset = self
            .compact_vectors
            .as_ref()
            .map(|mvs| self.builder.create_vector(mvs));

        // Build DPacketPayload
        let d_packet = generated::DPacketPayload::create(
            &mut self.builder,
            &generated::DPacketPayloadArgs {
                ref_sequence: self.ref_sequence,
                motion_vectors: mvs_offset,
                motion_vectors_compact: compact_offset,
                global_motion: None,
                region_deltas: None,
                timestamp_ms: self.timestamp_ms,
            },
        );

        // Wrap in AspPacketPayload union
        let asp_packet = generated::AspPacketPayload::create(
            &mut self.builder,
            &generated::AspPacketPayloadArgs {
                payload_type: generated::AspPayloadUnion::DPacketPayload,
                payload: Some(d_packet.as_union_value()),
            },
        );

        self.builder.finish(asp_packet, Some("ASP1"));

        let data = self.builder.finished_data();
        if buffer.len() < data.len() {
            return Err(FlatBuffersError::BufferTooSmall {
                needed: data.len(),
                got: buffer.len(),
            });
        }

        buffer[..data.len()].copy_from_slice(data);
        Ok(data.len())
    }
}

// =============================================================================
// I-Packet Builder
// =============================================================================

/// Builder for I-Packet (Keyframe)
pub struct IPacketBuilder<'a> {
    builder: FlatBufferBuilder<'a>,
    width: u32,
    height: u32,
    fps: f32,
    quality: FbQualityLevel,
    timestamp_ms: u64,
}

impl<'a> IPacketBuilder<'a> {
    /// Create a new I-Packet builder
    pub fn new(width: u32, height: u32, fps: f32) -> Self {
        Self {
            builder: FlatBufferBuilder::with_capacity(4096),
            width,
            height,
            fps,
            quality: FbQualityLevel::Medium,
            timestamp_ms: 0,
        }
    }

    /// Set quality level
    pub fn quality(mut self, q: FbQualityLevel) -> Self {
        self.quality = q;
        self
    }

    /// Set timestamp in milliseconds
    pub fn timestamp_ms(mut self, ts: u64) -> Self {
        self.timestamp_ms = ts;
        self
    }

    /// Build the packet and return serialized bytes
    pub fn build(mut self) -> Vec<u8> {
        // Build IPacketPayload
        let i_packet = generated::IPacketPayload::create(
            &mut self.builder,
            &generated::IPacketPayloadArgs {
                width: self.width,
                height: self.height,
                fps: self.fps,
                quality: self.quality,
                global_palette: None,
                regions: None,
                animation: None,
                timestamp_ms: self.timestamp_ms,
            },
        );

        // Wrap in AspPacketPayload union
        let asp_packet = generated::AspPacketPayload::create(
            &mut self.builder,
            &generated::AspPacketPayloadArgs {
                payload_type: generated::AspPayloadUnion::IPacketPayload,
                payload: Some(i_packet.as_union_value()),
            },
        );

        self.builder.finish(asp_packet, Some("ASP1"));
        self.builder.finished_data().to_vec()
    }
}

// =============================================================================
// S-Packet Builder (Sync/Control)
// =============================================================================

/// Builder for S-Packet (Sync/Control)
pub struct SPacketBuilder<'a> {
    builder: FlatBufferBuilder<'a>,
    command: FbSyncCommand,
    timestamp_ms: u64,
}

impl<'a> SPacketBuilder<'a> {
    /// Create a new S-Packet builder
    pub fn new(command: FbSyncCommand) -> Self {
        Self {
            builder: FlatBufferBuilder::with_capacity(128),
            command,
            timestamp_ms: 0,
        }
    }

    /// Create a Ping packet
    pub fn ping() -> Self {
        Self::new(FbSyncCommand::Ping)
    }

    /// Create a Pong packet
    pub fn pong() -> Self {
        Self::new(FbSyncCommand::Pong)
    }

    /// Create a RequestKeyframe packet
    pub fn request_keyframe() -> Self {
        Self::new(FbSyncCommand::RequestKeyframe)
    }

    /// Create an EndOfStream packet
    pub fn end_of_stream() -> Self {
        Self::new(FbSyncCommand::EndOfStream)
    }

    /// Set timestamp in milliseconds
    pub fn timestamp_ms(mut self, ts: u64) -> Self {
        self.timestamp_ms = ts;
        self
    }

    /// Build the packet and return serialized bytes
    pub fn build(mut self) -> Vec<u8> {
        // Build SPacketPayload
        let s_packet = generated::SPacketPayload::create(
            &mut self.builder,
            &generated::SPacketPayloadArgs {
                command: self.command,
                data_type: generated::SyncDataUnion::NONE,
                data: None,
                timestamp_ms: self.timestamp_ms,
            },
        );

        // Wrap in AspPacketPayload union
        let asp_packet = generated::AspPacketPayload::create(
            &mut self.builder,
            &generated::AspPacketPayloadArgs {
                payload_type: generated::AspPayloadUnion::SPacketPayload,
                payload: Some(s_packet.as_union_value()),
            },
        );

        self.builder.finish(asp_packet, Some("ASP1"));
        self.builder.finished_data().to_vec()
    }
}

// =============================================================================
// Packet Readers (Zero-Copy Access)
// =============================================================================

/// Read and verify a FlatBuffers packet
///
/// Returns the root AspPacketPayload table for zero-copy access.
pub fn read_packet(bytes: &[u8]) -> FbResult<AspPacketPayload<'_>> {
    // Verify file identifier
    if bytes.len() < 8 {
        return Err(FlatBuffersError::InvalidFormat(
            "Buffer too small".to_string(),
        ));
    }

    // Get the root with verification
    let packet = flatbuffers::root::<AspPacketPayload>(bytes)
        .map_err(|e| FlatBuffersError::InvalidFormat(e.to_string()))?;

    Ok(packet)
}

/// Read a D-Packet from bytes (zero-copy)
///
/// # Example
///
/// ```rust,ignore
/// let d_packet = read_d_packet(&bytes)?;
/// let mvs = d_packet.motion_vectors().unwrap();
/// for mv in mvs.iter() {
///     println!("dx={}, dy={}", mv.dx(), mv.dy());
/// }
/// ```
pub fn read_d_packet(bytes: &[u8]) -> FbResult<DPacketPayload<'_>> {
    let packet = read_packet(bytes)?;

    if packet.payload_type() != generated::AspPayloadUnion::DPacketPayload {
        return Err(FlatBuffersError::InvalidFormat(format!(
            "Expected DPacketPayload, got {:?}",
            packet.payload_type()
        )));
    }

    packet
        .payload_as_dpacket_payload()
        .ok_or(FlatBuffersError::MissingField("payload"))
}

/// Read an I-Packet from bytes (zero-copy)
pub fn read_i_packet(bytes: &[u8]) -> FbResult<IPacketPayload<'_>> {
    let packet = read_packet(bytes)?;

    if packet.payload_type() != generated::AspPayloadUnion::IPacketPayload {
        return Err(FlatBuffersError::InvalidFormat(format!(
            "Expected IPacketPayload, got {:?}",
            packet.payload_type()
        )));
    }

    packet
        .payload_as_ipacket_payload()
        .ok_or(FlatBuffersError::MissingField("payload"))
}

/// Read an S-Packet from bytes (zero-copy)
pub fn read_s_packet(bytes: &[u8]) -> FbResult<SPacketPayload<'_>> {
    let packet = read_packet(bytes)?;

    if packet.payload_type() != generated::AspPayloadUnion::SPacketPayload {
        return Err(FlatBuffersError::InvalidFormat(format!(
            "Expected SPacketPayload, got {:?}",
            packet.payload_type()
        )));
    }

    packet
        .payload_as_spacket_payload()
        .ok_or(FlatBuffersError::MissingField("payload"))
}

/// Get packet type without full parsing
pub fn get_packet_type(bytes: &[u8]) -> FbResult<generated::AspPayloadUnion> {
    let packet = read_packet(bytes)?;
    Ok(packet.payload_type())
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Create a D-Packet with motion vectors (convenience function)
pub fn create_d_packet(ref_sequence: u32, mvs: &[RustMotionVector], timestamp_ms: u64) -> Vec<u8> {
    DPacketBuilder::new(ref_sequence)
        .motion_vectors(mvs)
        .timestamp_ms(timestamp_ms)
        .build()
}

/// Create an I-Packet (convenience function)
pub fn create_i_packet(width: u32, height: u32, fps: f32, timestamp_ms: u64) -> Vec<u8> {
    IPacketBuilder::new(width, height, fps)
        .timestamp_ms(timestamp_ms)
        .build()
}

/// Create a Ping packet (convenience function)
pub fn create_ping(timestamp_ms: u64) -> Vec<u8> {
    SPacketBuilder::ping().timestamp_ms(timestamp_ms).build()
}

/// Create a Pong packet (convenience function)
pub fn create_pong(timestamp_ms: u64) -> Vec<u8> {
    SPacketBuilder::pong().timestamp_ms(timestamp_ms).build()
}

// =============================================================================
// Builder Reuse API (Zero-Allocation Hot Loop)
// =============================================================================

/// Encode a D-Packet reusing an existing FlatBufferBuilder
///
/// This function is designed for hot loops where you want to avoid
/// per-frame allocations. The builder is reset at the start of each call.
///
/// # Optimization
/// - Uses `start_vector` + `push` pattern (no intermediate Vec allocation)
/// - FlatBuffers builds vectors backwards, so we iterate in reverse
///
/// # Example
///
/// ```rust,ignore
/// use flatbuffers::FlatBufferBuilder;
/// use libasp::flatbuffers_api::encode_d_packet_with_builder;
///
/// // Create builder once
/// let mut builder = FlatBufferBuilder::with_capacity(4096);
///
/// // Reuse in hot loop
/// for frame in frames {
///     let bytes = encode_d_packet_with_builder(&mut builder, frame.ref_seq, &frame.mvs, frame.ts);
///     socket.send(bytes)?;
/// }
/// ```
#[inline]
pub fn encode_d_packet_with_builder<'a>(
    builder: &'a mut FlatBufferBuilder<'static>,
    ref_sequence: u32,
    mvs: &[RustMotionVector],
    timestamp_ms: u64,
) -> &'a [u8] {
    builder.reset();

    // 1. Serialize Motion Vectors (Direct Push - Zero Allocation)
    // FlatBuffers builds vectors backwards, so we iterate in reverse
    let mvs_len = mvs.len();

    builder.start_vector::<FbMotionVector>(mvs_len);
    for i in (0..mvs_len).rev() {
        // SAFETY: i is always in bounds [0, mvs_len)
        let mv = unsafe { mvs.get_unchecked(i) };
        let fb_mv = FbMotionVector::new(mv.block_x, mv.block_y, mv.dx, mv.dy, mv.sad);
        builder.push(fb_mv);
    }
    let mvs_offset = builder.end_vector::<FbMotionVector>(mvs_len);

    // 2. Build DPacketPayload
    let d_packet = generated::DPacketPayload::create(
        builder,
        &generated::DPacketPayloadArgs {
            ref_sequence,
            motion_vectors: Some(mvs_offset),
            motion_vectors_compact: None,
            global_motion: None,
            region_deltas: None,
            timestamp_ms,
        },
    );

    // 3. Wrap in AspPacketPayload union
    let asp_packet = generated::AspPacketPayload::create(
        builder,
        &generated::AspPacketPayloadArgs {
            payload_type: generated::AspPayloadUnion::DPacketPayload,
            payload: Some(d_packet.as_union_value()),
        },
    );

    builder.finish(asp_packet, Some("ASP1"));
    builder.finished_data()
}

/// Encode a D-Packet with compact motion vectors reusing an existing builder
///
/// # Optimization
/// - Uses `start_vector` + `push` pattern (no intermediate Vec allocation)
#[inline]
pub fn encode_d_packet_compact_with_builder<'a>(
    builder: &'a mut FlatBufferBuilder<'static>,
    ref_sequence: u32,
    compact_mvs: &[(i8, i8)],
    timestamp_ms: u64,
) -> &'a [u8] {
    builder.reset();

    // Direct Push - Zero Allocation
    let mvs_len = compact_mvs.len();

    builder.start_vector::<FbMotionVectorCompact>(mvs_len);
    for i in (0..mvs_len).rev() {
        // SAFETY: i is always in bounds [0, mvs_len)
        let (dx, dy) = unsafe { *compact_mvs.get_unchecked(i) };
        let fb_mv = FbMotionVectorCompact::new(dx, dy);
        builder.push(fb_mv);
    }
    let mvs_offset = builder.end_vector::<FbMotionVectorCompact>(mvs_len);

    // Build DPacketPayload
    let d_packet = generated::DPacketPayload::create(
        builder,
        &generated::DPacketPayloadArgs {
            ref_sequence,
            motion_vectors: None,
            motion_vectors_compact: Some(mvs_offset),
            global_motion: None,
            region_deltas: None,
            timestamp_ms,
        },
    );

    // Wrap in AspPacketPayload union
    let asp_packet = generated::AspPacketPayload::create(
        builder,
        &generated::AspPacketPayloadArgs {
            payload_type: generated::AspPayloadUnion::DPacketPayload,
            payload: Some(d_packet.as_union_value()),
        },
    );

    builder.finish(asp_packet, Some("ASP1"));
    builder.finished_data()
}

/// Encode an I-Packet reusing an existing FlatBufferBuilder
#[inline]
pub fn encode_i_packet_with_builder<'a>(
    builder: &'a mut FlatBufferBuilder<'static>,
    width: u32,
    height: u32,
    fps: f32,
    quality: FbQualityLevel,
    timestamp_ms: u64,
) -> &'a [u8] {
    builder.reset();

    // Build IPacketPayload
    let i_packet = generated::IPacketPayload::create(
        builder,
        &generated::IPacketPayloadArgs {
            width,
            height,
            fps,
            quality,
            global_palette: None,
            regions: None,
            animation: None,
            timestamp_ms,
        },
    );

    // Wrap in AspPacketPayload union
    let asp_packet = generated::AspPacketPayload::create(
        builder,
        &generated::AspPacketPayloadArgs {
            payload_type: generated::AspPayloadUnion::IPacketPayload,
            payload: Some(i_packet.as_union_value()),
        },
    );

    builder.finish(asp_packet, Some("ASP1"));
    builder.finished_data()
}

/// Encode an S-Packet reusing an existing FlatBufferBuilder
#[inline]
pub fn encode_s_packet_with_builder<'a>(
    builder: &'a mut FlatBufferBuilder<'static>,
    command: FbSyncCommand,
    timestamp_ms: u64,
) -> &'a [u8] {
    builder.reset();

    // Build SPacketPayload
    let s_packet = generated::SPacketPayload::create(
        builder,
        &generated::SPacketPayloadArgs {
            command,
            data_type: generated::SyncDataUnion::NONE,
            data: None,
            timestamp_ms,
        },
    );

    // Wrap in AspPacketPayload union
    let asp_packet = generated::AspPacketPayload::create(
        builder,
        &generated::AspPacketPayloadArgs {
            payload_type: generated::AspPayloadUnion::SPacketPayload,
            payload: Some(s_packet.as_union_value()),
        },
    );

    builder.finish(asp_packet, Some("ASP1"));
    builder.finished_data()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_d_packet_roundtrip() {
        let mvs = vec![
            RustMotionVector::new(0, 0, 5, -3, 100),
            RustMotionVector::new(1, 0, 2, 1, 50),
            RustMotionVector::new(2, 0, -1, 0, 25),
        ];

        let bytes = DPacketBuilder::new(42)
            .motion_vectors(&mvs)
            .timestamp_ms(12345)
            .build();

        // Verify we can read it back
        let packet = read_d_packet(&bytes).unwrap();
        assert_eq!(packet.ref_sequence(), 42);
        assert_eq!(packet.timestamp_ms(), 12345);

        let read_mvs = packet.motion_vectors().unwrap();
        assert_eq!(read_mvs.len(), 3);

        let mv0 = read_mvs.get(0);
        assert_eq!(mv0.block_x(), 0);
        assert_eq!(mv0.block_y(), 0);
        assert_eq!(mv0.dx(), 5);
        assert_eq!(mv0.dy(), -3);
        assert_eq!(mv0.sad(), 100);
    }

    #[test]
    fn test_i_packet_roundtrip() {
        let bytes = IPacketBuilder::new(1920, 1080, 30.0)
            .quality(FbQualityLevel::High)
            .timestamp_ms(0)
            .build();

        let packet = read_i_packet(&bytes).unwrap();
        assert_eq!(packet.width(), 1920);
        assert_eq!(packet.height(), 1080);
        assert!((packet.fps() - 30.0).abs() < 0.01);
        assert_eq!(packet.quality(), FbQualityLevel::High);
    }

    #[test]
    fn test_s_packet_roundtrip() {
        let bytes = SPacketBuilder::ping().timestamp_ms(99999).build();

        let packet = read_s_packet(&bytes).unwrap();
        assert_eq!(packet.command(), FbSyncCommand::Ping);
        assert_eq!(packet.timestamp_ms(), 99999);
    }

    #[test]
    fn test_packet_type_detection() {
        let d_bytes = create_d_packet(1, &[], 0);
        assert_eq!(
            get_packet_type(&d_bytes).unwrap(),
            generated::AspPayloadUnion::DPacketPayload
        );

        let i_bytes = create_i_packet(640, 480, 30.0, 0);
        assert_eq!(
            get_packet_type(&i_bytes).unwrap(),
            generated::AspPayloadUnion::IPacketPayload
        );

        let s_bytes = create_ping(0);
        assert_eq!(
            get_packet_type(&s_bytes).unwrap(),
            generated::AspPayloadUnion::SPacketPayload
        );
    }

    #[test]
    fn test_build_into_buffer() {
        let mvs = vec![RustMotionVector::new(0, 0, 1, 1, 10)];

        let mut buffer = [0u8; 1024];
        let len = DPacketBuilder::new(1)
            .motion_vectors(&mvs)
            .build_into(&mut buffer)
            .unwrap();

        // Verify we can read from the buffer
        let packet = read_d_packet(&buffer[..len]).unwrap();
        assert_eq!(packet.ref_sequence(), 1);
    }

    #[test]
    fn test_compact_motion_vectors() {
        let compact = vec![(5i8, -3i8), (2, 1), (-1, 0)];

        let bytes = DPacketBuilder::new(1).compact_vectors(&compact).build();

        let packet = read_d_packet(&bytes).unwrap();
        let mvs = packet.motion_vectors_compact().unwrap();
        assert_eq!(mvs.len(), 3);

        let mv0 = mvs.get(0);
        assert_eq!(mv0.dx(), 5);
        assert_eq!(mv0.dy(), -3);
    }

    #[test]
    fn test_type_conversions() {
        let rust_mv = RustMotionVector::new(10, 20, -5, 3, 500);
        let fb_mv = motion_vector_to_fb(&rust_mv);
        let back = motion_vector_from_fb(&fb_mv);

        assert_eq!(rust_mv.block_x, back.block_x);
        assert_eq!(rust_mv.block_y, back.block_y);
        assert_eq!(rust_mv.dx, back.dx);
        assert_eq!(rust_mv.dy, back.dy);
        assert_eq!(rust_mv.sad, back.sad);
    }

    #[test]
    fn test_builder_reuse_d_packet() {
        let mut builder = FlatBufferBuilder::with_capacity(4096);

        // First encoding
        let mvs1 = vec![RustMotionVector::new(0, 0, 5, -3, 100)];
        let bytes1 = encode_d_packet_with_builder(&mut builder, 1, &mvs1, 1000);
        let bytes1_copy = bytes1.to_vec();

        // Second encoding (builder reused)
        let mvs2 = vec![
            RustMotionVector::new(1, 1, 2, 1, 50),
            RustMotionVector::new(2, 2, -1, 0, 25),
        ];
        let bytes2 = encode_d_packet_with_builder(&mut builder, 2, &mvs2, 2000);
        let bytes2_copy = bytes2.to_vec();

        // Verify both can be read correctly
        let packet1 = read_d_packet(&bytes1_copy).unwrap();
        assert_eq!(packet1.ref_sequence(), 1);
        assert_eq!(packet1.motion_vectors().unwrap().len(), 1);

        let packet2 = read_d_packet(&bytes2_copy).unwrap();
        assert_eq!(packet2.ref_sequence(), 2);
        assert_eq!(packet2.motion_vectors().unwrap().len(), 2);
    }

    #[test]
    fn test_builder_reuse_stress() {
        let mut builder = FlatBufferBuilder::with_capacity(1024);

        // Encode many packets with same builder
        for i in 0..100 {
            let mvs = vec![RustMotionVector::new(i as u16, 0, 1, 1, i)];
            let bytes = encode_d_packet_with_builder(&mut builder, i, &mvs, i as u64 * 10);

            let packet = read_d_packet(bytes).unwrap();
            assert_eq!(packet.ref_sequence(), i);
            assert_eq!(packet.timestamp_ms(), i as u64 * 10);
        }
    }

    #[test]
    fn test_encode_i_packet_with_builder() {
        let mut builder = FlatBufferBuilder::with_capacity(1024);

        let bytes = encode_i_packet_with_builder(
            &mut builder,
            1920,
            1080,
            30.0,
            FbQualityLevel::High,
            12345,
        );

        let packet = read_i_packet(bytes).unwrap();
        assert_eq!(packet.width(), 1920);
        assert_eq!(packet.height(), 1080);
        assert!((packet.fps() - 30.0).abs() < 0.01);
        assert_eq!(packet.quality(), FbQualityLevel::High);
    }

    #[test]
    fn test_encode_s_packet_with_builder() {
        let mut builder = FlatBufferBuilder::with_capacity(256);

        let bytes =
            encode_s_packet_with_builder(&mut builder, FbSyncCommand::RequestKeyframe, 99999);

        let packet = read_s_packet(bytes).unwrap();
        assert_eq!(packet.command(), FbSyncCommand::RequestKeyframe);
        assert_eq!(packet.timestamp_ms(), 99999);
    }
}
